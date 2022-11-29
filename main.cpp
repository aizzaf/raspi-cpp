#include <iostream>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

int main () {
  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile("tf_lite_quant_model.tflite");

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Interpreter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);

  // Allocate tensor buffers.
  interpreter->AllocateTensors();

  float* input = interpreter->typed_input_tensor<float>(0);
  input[0] = 0.1f;
  input[1] = 0.0f;
  input[2] = 0.0f;
  input[3] = 0.2f;
  input[4] = 0.0f;
  input[5] = 0.0f;
  input[6] = 0.1f;
  input[7] = 0.1f;
  input[8] = 0.2f;
  input[9] = 0.2f;
  input[10] = 0.1f;
  input[11] = 0.1f;
  input[12] = 0.2f;
  input[13] = 0.3f;
  input[14] = 0.4f;
  input[15] = 0.1f;

  // Run inference
  interpreter->Invoke();

  float* output = interpreter->typed_output_tensor<float>(0);
  std::cout << input[0] << std::endl;
  std::cout << output[39] << std::endl;
}
