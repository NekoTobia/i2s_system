package com.example.camera_x
import android.Manifest
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.media.MediaPlayer
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageCapture
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.core.Preview
import androidx.camera.core.CameraSelector
import android.util.Log
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import com.example.camera_x.databinding.ActivityMainBinding
import java.io.IOException
import java.nio.ByteBuffer
import java.util.Locale
import okhttp3.*
import java.util.concurrent.TimeUnit
import org.json.JSONObject
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer

typealias LumaListener = (luma: Double) -> Unit


class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var output: ImageCapture.OutputFileResults
    private var imageCapture: ImageCapture? = null
    private lateinit var okHttpClient: OkHttpClient
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var speechRecognizer: SpeechRecognizer
    private var isListening = false
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        speechRecognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                // Called when the recognizer is ready to listen
            }

            override fun onBeginningOfSpeech() {
                // Called when the user starts speaking
            }

            override fun onRmsChanged(rmsdB: Float) {
                // Called when the input volume changes
            }

            override fun onBufferReceived(buffer: ByteArray?) {
                // Called when audio data is received
            }

            override fun onEndOfSpeech() {
                // Called when the user stops speaking
            }

            override fun onError(error: Int) {
                // Called when an error occurs during recognition
            }

            override fun onResults(results: Bundle?) {
                // Called when speech recognition results are available
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!matches.isNullOrEmpty()) {
                    val command = matches[0].lowercase(Locale.getDefault())

                        // Start image capture when the voice command is recognized
                    val spokenText =  matches?.joinToString(" ") { it }?:""
                    sendTextToServer(spokenText)
                    takePhoto()


                }
            }

            override fun onPartialResults(partialResults: Bundle?) {
                // Called when partial recognition results are available
            }

            override fun onEvent(eventType: Int, params: Bundle?) {
                // Called when a network or other error occurs
            }
        })

        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)
        okHttpClient = OkHttpClient.Builder()
            .connectTimeout(60, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .build()
        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions()
        }

        // Set up the listeners for take photo and video capture buttons
        viewBinding.imageCaptureButton.setOnClickListener { takePhoto() }
        //viewBinding.videoCaptureButton.setOnClickListener { sendPhoto() }
        viewBinding.transparentButton.setOnClickListener{startListening()}

        cameraExecutor = Executors.newSingleThreadExecutor()

    }


    private fun takePhoto() {
        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return
        viewBinding.imageCaptureButton.isEnabled = false
        Toast.makeText(this, "正在處理照片...", Toast.LENGTH_SHORT).show()
        // Set up image capture listener, which is triggered after the photo has been taken
        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    // Convert the captured image to a byte array
                    val byteArray = imageToByteArray(image)

                    // Call a function to send the byte array to your backend
                    sendPhotoToBackend(byteArray)

                    // Close the ImageProxy
                    image.close()

                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exception.message}", exception)

                }
            }
        )
    }

    private fun sendPhotoToBackend(photoData: ByteArray) {
        val url = "http://Your ip address/upload_image" // Replace with your backend API URL
        val formBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("image", "image.png", RequestBody.create("image/png".toMediaTypeOrNull(), photoData))
            .build()

        val request = Request.Builder()
            .url(url)
            .post(formBody)
            .build()

        okHttpClient.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e(TAG, "Failed to send photo to backend: ${e.message}", e)
            }

            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    val responseBody = response.body?.string()
                    val jsonResponse = JSONObject(responseBody)

                    val audioUrl = jsonResponse.optString("audio_url")

                    if (audioUrl != null) {
                        runOnUiThread {
                            Toast.makeText(applicationContext, "圖片已送出", Toast.LENGTH_SHORT).show()

                            // Download and play the audio file
                            val mediaPlayer = MediaPlayer()
                            mediaPlayer.setDataSource("http://Your ip address/get_audio/$audioUrl")
                            mediaPlayer.setOnPreparedListener {
                                mediaPlayer.start()
                            }
                            mediaPlayer.prepareAsync()
                            viewBinding.imageCaptureButton.isEnabled = true


                        }
                    } else {
                        runOnUiThread {
                            Toast.makeText(applicationContext, "No audio URL received", Toast.LENGTH_SHORT).show()
                        }
                        startListening()
                    }
                } else {
                    runOnUiThread {
                        Toast.makeText(applicationContext, "Failed to send image", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        })
    }

    private fun imageToByteArray(image: ImageProxy): ByteArray {
        val buffer: ByteBuffer = image.planes[0].buffer
        val data = ByteArray(buffer.remaining())
        buffer.get(data)
        return data
    }

    private fun sendTextToServer(text: String) {
        val formbody: RequestBody = FormBody.Builder()
            .add("sample", text)
            .build()

        val request: Request = Request.Builder().url("http://Your ip address/text")
            .post(formbody)
            .build()

        okHttpClient!!.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    Toast.makeText(applicationContext, "Server down", Toast.LENGTH_SHORT).show()
                }
            }

            @Throws(IOException::class)
            override fun onResponse(call: Call, response: Response) {
                if (response.body!!.string() == "received") {
                    runOnUiThread {
                        Toast.makeText(applicationContext, "Data received", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        })
    }





    private fun startCamera() {
        imageCapture = ImageCapture.Builder().build()
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview,imageCapture)

            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun requestPermissions() {
        activityResultLauncher.launch(REQUIRED_PERMISSIONS)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
    private fun startListening() {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        intent.putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        speechRecognizer.startListening(intent)
        isListening= true
        viewBinding.imageCaptureButton.isEnabled = true
    }

    companion object {
        private const val TAG = "CameraXApp"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private val REQUIRED_PERMISSIONS =
            mutableListOf (
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO
            ).apply {
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                    add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                }
            }.toTypedArray()
    }
    private val activityResultLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions())
        { permissions ->
            // Handle Permission granted/rejected
            var permissionGranted = true
            permissions.entries.forEach {
                if (it.key in REQUIRED_PERMISSIONS && it.value == false)
                    permissionGranted = false
            }
            if (!permissionGranted) {
                Toast.makeText(baseContext,
                    "Permission request denied",
                    Toast.LENGTH_SHORT).show()
            } else {
                startCamera()
            }
        }


}