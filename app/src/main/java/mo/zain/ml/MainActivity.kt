package mo.zain.ml

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import mo.zain.ml.ml.Iris
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        var button:Button=findViewById(R.id.button)

        button.setOnClickListener(View.OnClickListener {
            var editText1:EditText=findViewById(R.id.editTextNumberDecimal)
            var editText2:EditText=findViewById(R.id.editTextNumberDecimal2)
            var editText3:EditText=findViewById(R.id.editTextNumberDecimal3)
            var editText4:EditText=findViewById(R.id.editTextNumberDecimal4)


            var value1:Float=editText1.text.toString().toFloat()
            var value2:Float=editText2.text.toString().toFloat()
            var value3:Float=editText3.text.toString().toFloat()
            var value4:Float=editText4.text.toString().toFloat()

            var byteBuffer:ByteBuffer= ByteBuffer.allocateDirect(4*4)
            byteBuffer.putFloat(value1)
            byteBuffer.putFloat(value2)
            byteBuffer.putFloat(value3)
            byteBuffer.putFloat(value4)

            val model = Iris.newInstance(this)

            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 4), DataType.FLOAT32)
            inputFeature0.loadBuffer(byteBuffer)

            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

            var textView:TextView=findViewById(R.id.textView)
            textView.setText(" Iris-Setosa - "+outputFeature0[0].toString()+"\n Iris-Versicolor - "+
                    outputFeature0[1].toString()+"\n Iris-Virginica - "+
                    outputFeature0[2].toString())

            model.close()
        })
    }
}