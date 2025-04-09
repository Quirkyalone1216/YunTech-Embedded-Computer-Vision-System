package com.example.androidpythonopencv;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.chaquo.python.Python;
import com.chaquo.python.PyObject;
import com.chaquo.python.android.AndroidPlatform;

import java.io.ByteArrayOutputStream;



public class MainActivity extends AppCompatActivity {
    Button Go_btn;
    ImageView src_image, res_image;

    BitmapDrawable drawable;
    Bitmap bitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Go_btn = findViewById(R.id.Go_button);
        src_image = (ImageView) findViewById(R.id.source_imageview);
        res_image = (ImageView) findViewById(R.id.response_imageview);
        //初始化python环境
        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }
        Python python_cv = Python.getInstance();

        Go_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // 获取源图片并转换为Bitmap对象
                drawable = (BitmapDrawable) src_image.getDrawable();
                bitmap = drawable.getBitmap();
                // 将Bitmap转换为byte[]对象
                ByteArrayOutputStream stream = new ByteArrayOutputStream();
                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream);
                byte[] byteArray = stream.toByteArray();

                // 调用Python方法处理图片
                PyObject cvObject = python_cv.getModule("opencv_python");
                Double tiltAngle = cvObject.callAttr("CalTiltAngle", byteArray).toJava(Double.class);

                // 使用 Logger 輸出傾斜角度
                Log.d("TiltAngle", "傾斜角度: " + tiltAngle + " 度");
            }
        });
    }
}

//    @Override
//    protected void onCreate(Bundle savedInstanceState) {
//        super.onCreate(savedInstanceState);
//        setContentView(R.layout.activity_main);
//
//        Log.d("DEBUG", "HELLO WORLD layout loaded");
//    }
