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

import androidx.annotation.NonNull;                                  // 新增
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;                  // 新增
import androidx.camera.core.ImageProxy;                            // 新增
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.camera.core.ImageAnalysis;

import com.chaquo.python.Python;
import com.chaquo.python.PyObject;
import com.chaquo.python.android.AndroidPlatform;
import com.google.common.util.concurrent.ListenableFuture;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;                                       // 新增
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    // Chaquopy Python 實例
    private Python py;

    // CameraX UI 元件 & 執行緒
    private PreviewView previewView;
    private ImageView resultView;
    private ImageCapture imageCapture;
    private ExecutorService cameraExecutor;

    // 權限啟動器
    private final ActivityResultLauncher<String> cameraPermLauncher =
            registerForActivityResult(
                    new ActivityResultContracts.RequestPermission(),
                    new ActivityResultCallback<Boolean>() {
                        @Override
                        public void onActivityResult(Boolean granted) {
                            if (granted) startCamera();
                            else Log.e("MainActivity", "Camera permission denied");
                        }
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

        previewView = findViewById(R.id.previewView);
        resultView  = findViewById(R.id.resultView);
        Button processBtn = findViewById(R.id.processBtn);

        cameraExecutor = Executors.newSingleThreadExecutor();
        processBtn.setOnClickListener(v -> captureAndProcess());

        if (ContextCompat.checkSelfPermission(
                this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            cameraPermLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    /** 啟動 CameraX Preview + ImageCapture **/
    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                // 1. Preview 用例
                Preview preview = new Preview.Builder()
                        .build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                // 2. ImageCapture 用例（原拍照功能）
                imageCapture = new ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                        .build();

                // 3. ImageAnalysis 用例：輸出 YUV_420_888，取得三平面影格
                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                        .build();
                analysis.setAnalyzer(cameraExecutor, image -> {
                    // 拿到三平面後轉 NV21
                    byte[] nv21 = imageProxyToNv21(image);
                    int w = image.getWidth();
                    int h = image.getHeight();
                    image.close();

                    // 背景執行 Python 處理，並在 UI Thread 更新畫面
                    cameraExecutor.execute(() -> {
                        try {
                            PyObject func   = py.getModule("opencv_python").get("process_nv21");
                            PyObject result = func.call(nv21, w, h);
                            byte[] outPng   = result.toJava(byte[].class);

                            runOnUiThread(() -> {
                                Bitmap bmp = BitmapFactory.decodeByteArray(outPng, 0, outPng.length);
                                resultView.setImageBitmap(bmp);
                                resultView.setVisibility(View.VISIBLE);
                                resultView.bringToFront();
                                resultView.setRotation(90f);
                            });
                        } catch (Exception e) {
                            Log.e("Python", "Error in Python call", e);
                        }
                    });
                });

                // 4. 使用後鏡頭
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build();

                // 5. 解除綁定再綁定所有 use-case
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

    /** 按下按鈕 → 拍照 → Python 處理 → 回傳顯示 **/
    private void captureAndProcess() {
        imageCapture.takePicture(
                cameraExecutor,
                new ImageCapture.OnImageCapturedCallback() {
                    @Override
                    public void onCaptureSuccess(@NonNull ImageProxy image) {
                        // 1) 轉 NV21 byte[]
                        byte[] nv21 = imageProxyToNv21(image);
                        int width  = image.getWidth();
                        int height = image.getHeight();
                        image.close();

                        // 2) 背景執行 Python
                        cameraExecutor.execute(() -> {
                            try {
                                PyObject func   = py.getModule("opencv_python").get("process_nv21");
                                PyObject result = func.call(nv21, width, height);
                                byte[] outPng   = result.toJava(byte[].class);

                                // 3) 切回 UI 更新
                                runOnUiThread(() -> {
                                    Bitmap outBmp = BitmapFactory.decodeByteArray(outPng, 0, outPng.length);
                                    resultView.setImageBitmap(outBmp);
                                    resultView.setVisibility(View.VISIBLE);
                                    resultView.bringToFront();
                                    resultView.setRotation(90f);
                                });
                            } catch (Exception e) {
                                Log.e("Python", "Error in Python call", e);
                            }
                        });
                    }

                    @Override
                    public void onError(@NonNull ImageCaptureException exc) {
                        Log.e("CameraX", "Capture error", exc);
                    }
                }
        );
    }

    /** 正確處理 rowStride / pixelStride，把 YUV_420_888 轉 NV21 **/
    private static byte[] imageProxyToNv21(ImageProxy image) {
        int w = image.getWidth();
        int h = image.getHeight();
        int ySize  = w * h;
        int uvSize = w * h / 2;               // NV21: ½ Y + interleaved VU
        byte[] nv21 = new byte[ySize + uvSize];

        // --------- Y plane ---------
        ImageProxy.PlaneProxy yPlane = image.getPlanes()[0];
        ByteBuffer yBuf      = yPlane.getBuffer();
        int yRowStride       = yPlane.getRowStride();
        int pos = 0;
        for (int row = 0; row < h; row++) {
            yBuf.position(row * yRowStride);
            yBuf.get(nv21, pos, w);
            pos += w;
        }

        // --------- UV planes ---------
        ImageProxy.PlaneProxy uPlane = image.getPlanes()[1];
        ImageProxy.PlaneProxy vPlane = image.getPlanes()[2];
        ByteBuffer uBuf      = uPlane.getBuffer();
        ByteBuffer vBuf      = vPlane.getBuffer();
        int rowStrideUV      = uPlane.getRowStride();     // same for both U & V
        int pixelStrideUV    = uPlane.getPixelStride();   // usually = 2

        // NV21 ordering is V first, then U
        for (int row = 0; row < h / 2; row++) {
            int rowStart = row * rowStrideUV;
            for (int col = 0; col < w / 2; col++) {
                int uIndex = rowStart + col * pixelStrideUV;
                int vIndex = rowStart + col * pixelStrideUV;
                nv21[pos++] = vBuf.get(vIndex);  // V
                nv21[pos++] = uBuf.get(uIndex);  // U
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
