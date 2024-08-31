package io.iron.notaaikotlin

import android.os.Bundle
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Canvas
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import android.widget.ImageView
import android.widget.Toast
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.io.FileInputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var tflite: Interpreter
    private lateinit var previewView: PreviewView
    private lateinit var imageView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        imageView = findViewById(R.id.imageView)

        cameraExecutor = Executors.newSingleThreadExecutor()

        tflite = Interpreter(loadModelFile("NP-converted-people_detection.tflite"))

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, ImageAnalyzer())
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private inner class ImageAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(imageProxy: ImageProxy) {
            val bitmap = imageProxy.toBitmap()

            val resultBitmap = bitmap?.let { runModel(it) }

            runOnUiThread {
                imageView.setImageBitmap(resultBitmap)
            }

            imageProxy.close()
        }

        private fun ImageProxy.toBitmap(): Bitmap? {
            val yBuffer = planes[0].buffer
            val uBuffer = planes[1].buffer
            val vBuffer = planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)

            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
            val imageBytes = out.toByteArray()
            val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

            if (bitmap == null) {
                Log.e("toBitmap", "Failed to decode YUV_420_888 to bitmap.")
                return null
            }

            val rotationDegrees = imageInfo.rotationDegrees
            if (rotationDegrees != 0) {
                val matrix = Matrix()
                matrix.postRotate(rotationDegrees.toFloat())
                return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            }

            return bitmap
        }

        private fun runModel(bitmap: Bitmap): Bitmap {
            val inputShape = tflite.getInputTensor(0).shape()
            val inputWidth = inputShape[2]
            val inputHeight = inputShape[1]

            val paddedBitmap = resizeAndPadBitmap(bitmap, inputWidth, inputHeight)

            val inputBuffer = convertBitmapToByteBuffer(paddedBitmap)

            val outputShape = tflite.getOutputTensor(0).shape()
            val output = Array(outputShape[0]) {
                Array(outputShape[1]) {
                    Array(outputShape[2]) {
                        Array(outputShape[3]) {
                            FloatArray(outputShape[4])
                        }
                    }
                }
            }

            tflite.run(inputBuffer, output)

            return processOutput(output, bitmap, paddedBitmap)
        }

        private fun resizeAndPadBitmap(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
            val aspectRatio = bitmap.width.toFloat() / bitmap.height.toFloat()
            val newWidth: Int
            val newHeight: Int

            if (aspectRatio > 1) {
                newWidth = targetWidth
                newHeight = (targetWidth / aspectRatio).toInt()
            } else {
                newWidth = (targetHeight * aspectRatio).toInt()
                newHeight = targetHeight
            }

            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)

            val paddedBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(paddedBitmap)
            canvas.drawColor(Color.BLACK)
            canvas.drawBitmap(scaledBitmap, (targetWidth - newWidth) / 2f, (targetHeight - newHeight) / 2f, null)

            return paddedBitmap
        }

        private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
            val inputShape = tflite.getInputTensor(0).shape()
            val inputSize = inputShape[1]
            val buffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
            buffer.order(ByteOrder.nativeOrder())

            val intValues = IntArray(inputSize * inputSize)

            bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

            for (pixelValue in intValues) {
                buffer.putFloat(((pixelValue shr 16) and 0xFF) / 255.0f)
                buffer.putFloat(((pixelValue shr 8) and 0xFF) / 255.0f)
                buffer.putFloat((pixelValue and 0xFF) / 255.0f)
            }

            buffer.rewind()
            return buffer
        }

        private fun processOutput(output: Array<Array<Array<Array<FloatArray>>>>, originalBitmap: Bitmap, paddedBitmap: Bitmap): Bitmap {
            val mutableBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(mutableBitmap)
            val paint = Paint().apply {
                color = Color.RED
                style = Paint.Style.STROKE
                strokeWidth = 8.0f
            }

            val scaleX = originalBitmap.width.toFloat() / paddedBitmap.width
            val scaleY = originalBitmap.height.toFloat() / paddedBitmap.height

            val centerX = originalBitmap.width / 2
            val centerY = originalBitmap.height / 2

            for (i in output.indices) {
                for (j in output[i].indices) {
                    for (k in output[i][j].indices) {
                        for (l in output[i][j][k].indices) {
                            val data = output[i][j][k][l]
                            if (data[4] > 0.5) {
                                var left = (data[0] * paddedBitmap.width) * scaleX - centerX
                                var top = (data[1] * paddedBitmap.height) * scaleY - centerY
                                var right = (left + data[2] * paddedBitmap.width) * scaleX - centerX
                                var bottom = (top + data[3] * paddedBitmap.height) * scaleY - centerY

                                left = left.coerceIn(-centerX.toFloat(), centerX.toFloat())
                                top = top.coerceIn(-centerY.toFloat(), centerY.toFloat())
                                right = right.coerceIn(-centerX.toFloat(), centerX.toFloat())
                                bottom = bottom.coerceIn(-centerY.toFloat(), centerY.toFloat())

                                left += centerX
                                top += centerY
                                right += centerX
                                bottom += centerY

                                Log.d("BoundingBox", "Clipped Left: $left, Top: $top, Right: $right, Bottom: $bottom")

                                canvas.drawRect(left, top, right, bottom, paint)
                            }
                        }
                    }
                }
            }

            return mutableBitmap
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(
                    this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "CameraXApp"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}