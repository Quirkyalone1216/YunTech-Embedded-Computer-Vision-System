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
import android.graphics.ImageFormat;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.camera.core.Preview;
import androidx.camera.view.PreviewView;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.google.common.util.concurrent.ListenableFuture;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import android.content.res.AssetManager;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {
    // Chaquopy Python 實例
    private Python py;

    // 只保留這兩個 UI 元件
    private PreviewView previewView;
    private ImageView resultView;
    private Button processBtn;

    // CameraX 用例
    private ImageCapture imageCapture;
    private ExecutorService cameraExecutor;
    private volatile boolean isProcessing = false;
    // 控制即時辨識開關
    private volatile boolean realtimeEnabled = false;

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

        // 取得 PreviewView、resultView 與 processBtn
        previewView = findViewById(R.id.previewView);
        resultView  = findViewById(R.id.resultView);
        processBtn  = findViewById(R.id.processBtn);

        cameraExecutor = Executors.newSingleThreadExecutor();
        // 按下 Process 按鈕：切換即時辨識開關
        processBtn.setOnClickListener(v -> {
            realtimeEnabled = !realtimeEnabled;
            if (realtimeEnabled) {
                processBtn.setText("Stop");
                resultView.setVisibility(View.VISIBLE);
                Log.d("MainActivity", "Realtime recognition started");
            } else {
                processBtn.setText("Process");
                Log.d("MainActivity", "Realtime recognition stopped");
            }
        });

        // 初始隱藏 resultView，以免遮住預覽
        resultView.setVisibility(View.GONE);

        // 複製並初始化模型
        copyModelToInternalIfNeeded();
        String modelPath = new File(getFilesDir(), "face_model.tflite").getAbsolutePath();
        try {
            PyObject module = py.getModule("age_gender");
            module.callAttr("init", modelPath);
            Log.d("MainActivity", "Initialized Python model at: " + modelPath);
        } catch (Exception e) {
            Log.e("MainActivity", "Python model init failed", e);
        }

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

                // 1. Preview 用例：顯示預覽
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                // 2. ImageCapture 用例（觸發拍照用）
                imageCapture = new ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                        .build();

                // 3. ImageAnalysis 用例：取得 YUV_420_888 → 自動呼 analyseImage
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
                        preview,
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
        // 檢查格式：只處理 YUV_420_888 格式
        if (image.getFormat() != ImageFormat.YUV_420_888) {
            // 非 YUV 格式跳過
            image.close();
            return;
        }
        // 若未啟用即時辨識，跳過
        // Debug: 紀錄每幀呼叫
        Log.d("MainActivity", "analyseImage called; realtimeEnabled=" + realtimeEnabled);
        if (!realtimeEnabled) {
            image.close();
            return;
        }
        // 節流：若上一張尚在處理中，跳過
        if (isProcessing) {
            image.close();
            return;
        }
        isProcessing = true;

        // 先取 NV21 bytes
        byte[] nv21 = imageProxyToNv21(image);
        int width  = image.getWidth();
        int height = image.getHeight();
        image.close();

        // 背景執行 Python
        cameraExecutor.execute(() -> {
            try {
                PyObject module = py.getModule("age_gender");
                Log.d("MainActivity", "Calling Python process_realtime");
                PyObject result = module.callAttr("process_realtime", nv21, width, height);
                byte[] outPng   = result.toJava(byte[].class);

                runOnUiThread(() -> {
                    Bitmap outBmp = BitmapFactory.decodeByteArray(outPng, 0, outPng.length);
                    resultView.setImageBitmap(outBmp);
                    resultView.setVisibility(View.VISIBLE);
                    // 處理完成，允許下一次處理
                    isProcessing = false;
                    Log.d("MainActivity", "Displayed processed frame");
                });
            } catch (Exception e) {
                Log.e("Python", "Error in Python call", e);
                isProcessing = false;
                Toast.makeText(MainActivity.this, "Python error: " + e.getMessage(), Toast.LENGTH_SHORT).show();
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
                        // Capture 回傳的 ImageProxy 格式可能非 YUV_420_888（如 JPEG），此時跳過或另行處理
                        if (image.getFormat() != ImageFormat.YUV_420_888) {
                            analyseImage(image);
                        } else {
                            // 若需要對拍照結果做處理，可改用不同方法處理 JPEG 格式
                            image.close();
                        }
                    }
                    @Override public void onError(@NonNull ImageCaptureException exc) {
                        Log.e("CameraX", "Capture error", exc);
                    }
                }
        );
    }

    /** Helper：把 ImageProxy 三平面轉 NV21 排序 **/
    private static byte[] imageProxyToNv21(ImageProxy image) {
        // 確保是 YUV_420_888，否則不應呼此方法
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        if (planes == null || planes.length < 3) {
            throw new IllegalArgumentException("imageProxyToNv21 requires YUV_420_888 format");
        }
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

        // UV 平面 (NV21 = VU interleaved)，使用 U/V plane 各自的 rowStride & pixelStride
        ByteBuffer uBuf = planes[1].getBuffer();
        ByteBuffer vBuf = planes[2].getBuffer();
        int rowStrideU   = planes[1].getRowStride();
        int pixelStrideU = planes[1].getPixelStride();
        int rowStrideV   = planes[2].getRowStride();
        int pixelStrideV = planes[2].getPixelStride();

        int pos = ySize;
        for (int row = 0; row < h/2; row++) {
            int rowStartU = row * rowStrideU;
            int rowStartV = row * rowStrideV;
            for (int col = 0; col < w/2; col++) {
                int uIndex = rowStartU + col * pixelStrideU;
                int vIndex = rowStartV + col * pixelStrideV;
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


    /** 複製 assets 中的 face_model.tflite 到內部私有 storage（如不存在才複製） **/
    private void copyModelToInternalIfNeeded() {
        File internalFile = new File(getFilesDir(), "face_model.tflite");
        if (internalFile.exists()) return;
        AssetManager assetManager = getAssets();
        try (InputStream is = assetManager.open("face_model.tflite");
             FileOutputStream os = openFileOutput("face_model.tflite", MODE_PRIVATE)) {
            byte[] buffer = new byte[4096];
            int read;
            while ((read = is.read(buffer)) != -1) {
                os.write(buffer, 0, read);
            }
            os.flush();
            Log.d("MainActivity", "Copied face_model.tflite to internal storage");
        } catch (IOException e) {
            Log.e("MainActivity", "Failed to copy face_model.tflite", e);
        }
    }
}
