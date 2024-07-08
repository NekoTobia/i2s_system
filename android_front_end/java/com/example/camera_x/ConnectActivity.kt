package com.example.camera_x

import android.content.Intent
import android.os.Bundle
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import okhttp3.Call
import okhttp3.Callback
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import java.io.IOException
import java.util.concurrent.TimeUnit

class ConnectActivity : AppCompatActivity() {
    // declare attribute for textview
    private var pagenameTextView: TextView? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_connect)
        pagenameTextView = findViewById(R.id.pagename)

        // creating a client
        val okHttpClient = OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS) // Set your desired connection timeout in seconds
            .readTimeout(30, TimeUnit.SECONDS) // Set your desired read timeout in seconds
            .build()

        // building a request
        val request: Request = Request.Builder().url("http://Your ip address").build()

        // making call asynchronously
        okHttpClient.newCall(request).enqueue(object : Callback {
            // called if server is unreachable
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    Toast.makeText(this@ConnectActivity, "Server is down", Toast.LENGTH_SHORT).show()
                    pagenameTextView!!.text = "Error connecting to the server"
                }
            }

            @Throws(IOException::class)  // called if we get a
            // response from the server
            override fun onResponse(
                call: Call,
                response: Response
            ) {
                if (response.isSuccessful) {
                    // Start the DummyActivity if the server is reachable
                    val intent = Intent(this@ConnectActivity, MainActivity::class.java)
                    startActivity(intent)
                    finish()
                } else {
                    runOnUiThread {
                        Toast.makeText(this@ConnectActivity, "Server responded with an error", Toast.LENGTH_SHORT).show()
                        pagenameTextView!!.text = "Server responded with an error"
                    }
                }
            }
        })
    }
}
