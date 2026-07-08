package com.example.aura_nav

import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothSocket
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.IOException
import java.io.OutputStream
import java.nio.charset.StandardCharsets
import java.util.concurrent.Executors
import java.util.UUID

class MainActivity : FlutterActivity() {
    private val channelName = "aura_nav/bluetooth"
    private val sppUuid: UUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB")

    @Volatile
    private var bluetoothSocket: BluetoothSocket? = null

    @Volatile
    private var bluetoothOutput: OutputStream? = null

    private val bluetoothIoExecutor = Executors.newSingleThreadExecutor()

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            channelName,
        ).setMethodCallHandler { call, result ->
            when (call.method) {
                "isSupported" -> result.success(bluetoothAdapter() != null)
                "isEnabled" -> result.success(bluetoothAdapter()?.isEnabled == true)
                "getBondedDevices" -> result.success(getBondedDevices())
                "connect" -> connect(call, result)
                "disconnect" -> {
                    disconnectSocket()
                    result.success(null)
                }
                "sendPayload" -> sendPayload(call, result)
                else -> result.notImplemented()
            }
        }
    }

    @Suppress("DEPRECATION")
    private fun bluetoothAdapter(): BluetoothAdapter? {
        val manager = getSystemService(BLUETOOTH_SERVICE) as? BluetoothManager
        return manager?.adapter ?: BluetoothAdapter.getDefaultAdapter()
    }

    @SuppressLint("MissingPermission")
    private fun getBondedDevices(): List<Map<String, String>> {
        val adapter = bluetoothAdapter() ?: return emptyList()
        return adapter.bondedDevices
            ?.sortedBy { device -> device.name ?: device.address }
            ?.map { device ->
                mapOf(
                    "name" to (device.name ?: device.address),
                    "address" to device.address,
                )
            }
            ?: emptyList()
    }

    @SuppressLint("MissingPermission")
    private fun connect(call: MethodCall, result: MethodChannel.Result) {
        val address = call.argument<String>("address")?.trim()?.uppercase()
        if (address.isNullOrEmpty()) {
            result.error("INVALID_ADDRESS", "Bluetooth MAC address is required.", null)
            return
        }

        val adapter = bluetoothAdapter()
        if (adapter == null) {
            result.error("UNSUPPORTED", "This Android device does not support Bluetooth.", null)
            return
        }
        if (!adapter.isEnabled) {
            result.error("DISABLED", "Bluetooth is disabled on the Android device.", null)
            return
        }

        Thread {
            try {
                disconnectSocket()
                adapter.cancelDiscovery()

                val device = adapter.getRemoteDevice(address)
                val socket = connectWithFallback(device)
                bluetoothSocket = socket
                bluetoothOutput = socket.outputStream
                runOnUiThread { result.success(null) }
            } catch (e: Exception) {
                disconnectSocket()
                runOnUiThread {
                    result.error(
                        "CONNECT_FAILED",
                        e.message ?: "Unable to connect to the Bluetooth device.",
                        null,
                    )
                }
            }
        }.start()
    }

    @SuppressLint("MissingPermission")
    @Throws(IOException::class)
    private fun connectWithFallback(device: BluetoothDevice): BluetoothSocket {
        val secureSocket = device.createRfcommSocketToServiceRecord(sppUuid)
        try {
            secureSocket.connect()
            return secureSocket
        } catch (secureError: IOException) {
            try {
                secureSocket.close()
            } catch (_: IOException) {
            }

            try {
                val insecureSocket = device.createInsecureRfcommSocketToServiceRecord(sppUuid)
                insecureSocket.connect()
                return insecureSocket
            } catch (insecureError: IOException) {
                val channelOneSocket = createChannelOneSocket(device)
                channelOneSocket.connect()
                return channelOneSocket
            }
        }
    }

    @Throws(IOException::class)
    private fun createChannelOneSocket(device: BluetoothDevice): BluetoothSocket {
        try {
            val method = device.javaClass.getMethod("createRfcommSocket", Int::class.javaPrimitiveType)
            return method.invoke(device, 1) as BluetoothSocket
        } catch (e: Exception) {
            throw IOException("Unable to create fallback RFCOMM channel 1 socket.", e)
        }
    }

    private fun sendPayload(call: MethodCall, result: MethodChannel.Result) {
        val payload = call.argument<String>("payload") ?: ""
        bluetoothIoExecutor.execute {
            try {
                val output = bluetoothOutput
                if (output == null) {
                    runOnUiThread {
                        result.error("NOT_CONNECTED", "Bluetooth socket is not connected.", null)
                    }
                    return@execute
                }
                val framedPayload =
                    if (payload.endsWith("\n")) payload else "$payload\n"
                output.write(framedPayload.toByteArray(StandardCharsets.UTF_8))
                output.flush()
                runOnUiThread { result.success(null) }
            } catch (e: Exception) {
                disconnectSocket()
                runOnUiThread {
                    result.error(
                        "SEND_FAILED",
                        e.message ?: "Failed to send data to the Bluetooth device.",
                        null,
                    )
                }
            }
        }
    }

    @Synchronized
    private fun disconnectSocket() {
        try {
            bluetoothOutput?.close()
        } catch (_: IOException) {
        } finally {
            bluetoothOutput = null
        }

        try {
            bluetoothSocket?.close()
        } catch (_: IOException) {
        } finally {
            bluetoothSocket = null
        }
    }
}
