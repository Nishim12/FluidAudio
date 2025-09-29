#!/usr/bin/env swift

import Foundation
import CoreML

print("🍎 CoreML Model Compiler & Benchmark")
print("   Compiling and testing overlapped speech detection model")
print("")

let modelPath = "/Users/nishim.singhi/ml_env/FluidAudio/overlapped_speech_detection.mlmodel"
let compiledModelDir = "/Users/nishim.singhi/ml_env/FluidAudio/overlapped_speech_detection.mlmodelc"

// Check if model exists
guard FileManager.default.fileExists(atPath: modelPath) else {
    print("❌ Model not found at: \(modelPath)")
    exit(1)
}

// Compile the model
print("🔄 Compiling CoreML model...")
do {
    let modelURL = URL(fileURLWithPath: modelPath)
    let compiledURL = try MLModel.compileModel(at: modelURL)

    print("✅ Model compiled successfully!")
    print("   Compiled to: \(compiledURL.path)")

    // Move to expected location
    let destinationURL = URL(fileURLWithPath: compiledModelDir)
    if FileManager.default.fileExists(atPath: compiledModelDir) {
        try FileManager.default.removeItem(at: destinationURL)
    }
    try FileManager.default.moveItem(at: compiledURL, to: destinationURL)
    print("   Moved to: \(compiledModelDir)")

} catch {
    print("❌ Failed to compile model: \(error)")
    exit(1)
}

// Now test the compiled model
print("\n🧪 Testing compiled model...")
do {
    let compiledURL = URL(fileURLWithPath: compiledModelDir)
    let model = try MLModel(contentsOf: compiledURL)

    print("✅ Compiled model loads successfully!")

    // Print model information
    let description = model.modelDescription
    print("\n📊 Model Information:")
    print("   Inputs: \(description.inputDescriptionsByName.count)")
    print("   Outputs: \(description.outputDescriptionsByName.count)")

    for (name, inputDesc) in description.inputDescriptionsByName {
        print("   Input '\(name)': \(inputDesc.type)")
    }

    for (name, outputDesc) in description.outputDescriptionsByName {
        print("   Output '\(name)': \(outputDesc.type)")
    }

    // Quick inference test with dummy data
    var testInferenceTime: Double = 0.0
    if let inputName = description.inputDescriptionsByName.keys.first {
        print("\n⚡ Running quick inference test...")

        // Create dummy input (assuming typical mel-spectrogram shape)
        let dummyInput = try MLMultiArray(shape: [500, 80], dataType: .float32)
        let inputCount = 500 * 80
        let dataPointer = dummyInput.dataPointer.bindMemory(to: Float32.self, capacity: inputCount)

        // Fill with random data
        for i in 0..<inputCount {
            dataPointer[i] = Float32.random(in: -1.0...1.0)
        }

        let inputDict = [inputName: dummyInput]

        let startTime = CFAbsoluteTimeGetCurrent()
        let prediction = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputDict))
        testInferenceTime = CFAbsoluteTimeGetCurrent() - startTime

        print("✅ Inference successful!")
        print("   Inference time: \(String(format: "%.2f", testInferenceTime * 1000)) ms")

        // Show output info
        if let outputName = description.outputDescriptionsByName.keys.first,
           let outputValue = prediction.featureValue(for: outputName),
           let outputArray = outputValue.multiArrayValue {
            print("   Output shape: \(outputArray.shape)")
            print("   Output count: \(outputArray.count)")

            // Show a few output values
            if outputArray.count > 0 {
                let outputPointer = outputArray.dataPointer.bindMemory(to: Float32.self, capacity: outputArray.count)
                let sampleSize = min(5, outputArray.count)
                let samples = Array(UnsafeBufferPointer(start: outputPointer, count: sampleSize))
                print("   Sample outputs: \(samples.map { String(format: "%.4f", $0) }.joined(separator: ", "))")
            }
        }
    }

    print("\n🎉 Model compilation and testing completed successfully!")
    print("   You can now run benchmarks with the compiled model at:")
    print("   \(compiledModelDir)")

    // Provide next steps
    print("\n📝 Next Steps:")
    print("   1. The model is now compiled and ready for use")
    print("   2. You can integrate this into your FluidAudio project")
    print("   3. Use standard CoreML APIs for inference in production")
    print("   4. Expected performance: ~\(String(format: "%.2f", testInferenceTime * 1000))ms per inference")

} catch {
    print("❌ Failed to test compiled model: \(error)")
    exit(1)
}