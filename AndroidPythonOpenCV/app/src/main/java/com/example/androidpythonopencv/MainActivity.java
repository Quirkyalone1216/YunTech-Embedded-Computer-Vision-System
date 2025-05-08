package com.example.androidpythonopencv;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.google.common.util.concurrent.ListenableFuture;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    // Chaquopy Python 實例
    private Python py;

    // 只保留這兩個 UI 元件
    private ImageView resultView;
    private Button processBtn;

    // CameraX 用例
    private ImageCapture imageCapture;
    private ExecutorService cameraExecutor;

    // 動態請求 CAMERA 權限
    private final ActivityResultLauncher<String> cameraPermLauncher =
            registerForActivityResult(
                    new ActivityResultContracts.RequestPermission(),
                    granted -> {
                        if (granted) startCamera();
                        else Log.e("MainActivity", "Camera permission denied");
                    }
            );

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // 啟動 Chaquopy
        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }
        py = Python.getInstance();

        setContentView(R.layout.activity_main);

        // 移除 previewView，只綁定 resultView 和 processBtn
        resultView  = findViewById(R.id.resultView);
        processBtn  = findViewById(R.id.processBtn);

        cameraExecutor = Executors.newSingleThreadExecutor();
        processBtn.setOnClickListener(v -> captureAndProcess());

        // 動態權限
        if (ContextCompat.checkSelfPermission(
                this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            cameraPermLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    /** 啟動 CameraX，只綁定 ImageCapture + ImageAnalysis **/
    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                // ImageCapture 用例（觸發拍照用）
                imageCapture = new ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                        .build();

                // ImageAnalysis 用例：取得 YUV_420_888 → 自動呼 process_nv21
                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                        .build();
                analysis.setAnalyzer(cameraExecutor, this::analyseImage);

                // 選後鏡頭
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build();

                // 解除所有用例再綁定：只綁定 imageCapture、analysis
                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(
                        this,
                        cameraSelector,
                        imageCapture,
                        analysis
                );
            } catch (ExecutionException | InterruptedException e) {
                Log.e("MainActivity", "startCamera error", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    /** ImageAnalysis 回呼 → 轉 NV21 → Python 處理 → 顯示 **/
    private void analyseImage(ImageProxy image) {
        // 先取 NV21 bytes
        byte[] nv21 = imageProxyToNv21(image);
        int width  = image.getWidth();
        int height = image.getHeight();
        image.close();

        // 背景執行 Python
        cameraExecutor.execute(() -> {
            try {
                PyObject func   = py.getModule("rps_rock_android").get("process_nv21");
                PyObject result = func.call(nv21, width, height);
                byte[] outPng   = result.toJava(byte[].class);

                runOnUiThread(() -> {
                    Bitmap outBmp = BitmapFactory.decodeByteArray(outPng, 0, outPng.length);
                    resultView.setImageBitmap(outBmp);
                    resultView.setVisibility(View.VISIBLE);
                });
            } catch (Exception e) {
                Log.e("Python", "Error in Python call", e);
            }
        });
    }

    /** 按鈕觸發：執行一次拍照（可選） **/
    private void captureAndProcess() {
        if (imageCapture == null) return;

        imageCapture.takePicture(
                ContextCompat.getMainExecutor(this),
                new ImageCapture.OnImageCapturedCallback() {
                    @Override public void onCaptureSuccess(@NonNull ImageProxy image) {
                        analyseImage(image);
                    }
                    @Override public void onError(@NonNull ImageCaptureException exc) {
                        Log.e("CameraX", "Capture error", exc);
                    }
                }
        );
    }

    /** Helper：把 ImageProxy 三平面轉 NV21 排序 **/
    private static byte[] imageProxyToNv21(ImageProxy image) {
        int w = image.getWidth(), h = image.getHeight();
        int ySize  = w * h;
        int uvSize = w * h / 2;
        byte[] nv21 = new byte[ySize + uvSize];

        // Y 平面
        ByteBuffer yBuf = image.getPlanes()[0].getBuffer();
        int rowStrideY = image.getPlanes()[0].getRowStride();
        for (int row = 0; row < h; row++) {
            yBuf.position(row * rowStrideY);
            yBuf.get(nv21, row * w, w);
        }

        // UV 平面 (NV21 = VU interleaved)
        ByteBuffer uBuf       = image.getPlanes()[1].getBuffer();
        ByteBuffer vBuf       = image.getPlanes()[2].getBuffer();
        int rowStrideUV       = image.getPlanes()[1].getRowStride();
        int pixelStrideUV     = image.getPlanes()[1].getPixelStride();

        int pos = ySize;
        for (int row = 0; row < h/2; row++) {
            int rowStart = row * rowStrideUV;
            for (int col = 0; col < w/2; col++) {
                int uIndex = rowStart + col * pixelStrideUV;
                int vIndex = rowStart + col * pixelStrideUV;
                nv21[pos++] = vBuf.get(vIndex);
                nv21[pos++] = uBuf.get(uIndex);
            }
        }
        return nv21;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
    }
}
