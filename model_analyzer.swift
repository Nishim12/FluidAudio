#!/usr/bin/env swift

import Foundation
import CoreML

print("🔍 CoreML Model Analysis")
print("   Analyzing overlapped_speech_detection.mlmodelc")
print("")

class ModelAnalyzer {
    let modelPath = "/Users/nishim.singhi/ml_env/FluidAudio/overlapped_speech_detection.mlmodelc"

    func analyze() throws {
        print("🔄 Loading model...")
        let model = try MLModel(contentsOf: URL(fileURLWithPath: modelPath))

        print("✅ Model loaded successfully")
        print("")

        // Analyze model description
        let description = model.modelDescription

        print("📊 MODEL ARCHITECTURE ANALYSIS")
        print(String(repeating: "=", count: 50))

        // Input analysis
        print("\n🔹 Input Features:")
        for (name, feature) in description.inputDescriptionsByName {
            print("   \(name):")
            print("     Type: \(feature.type)")

            if let multiArray = feature.multiArrayConstraint {
                print("     Shape: \(multiArray.shape)")
                print("     Data Type: \(multiArray.dataType)")
            }

            print("     Optional: \(feature.isOptional)")
        }

        // Output analysis
        print("\n🔹 Output Features:")
        for (name, feature) in description.outputDescriptionsByName {
            print("   \(name):")
            print("     Type: \(feature.type)")

            if let multiArray = feature.multiArrayConstraint {
                print("     Shape: \(multiArray.shape)")
                print("     Data Type: \(multiArray.dataType)")
            }
        }

        // Metadata analysis
        print("\n🔹 Model Metadata:")
        let metadata = description.metadata
        for (key, value) in metadata {
            print("   \(key): \(value)")
        }

        // Performance characteristics
        print("\n🔹 Performance Characteristics:")
        print("   Input: Raw audio [1, 1, 32000] (2 seconds at 16kHz)")
        print("   Expected output: Overlap probability per frame")
        print("   Current DER: 48.2% (needs improvement to 20-25%)")

        // Model file size
        let fileURL = URL(fileURLWithPath: modelPath)
        if let attributes = try? FileManager.default.attributesOfItem(atPath: fileURL.path),
           let fileSize = attributes[.size] as? Int64 {
            let sizeInMB = Double(fileSize) / (1024 * 1024)
            print("   Model size: \(String(format: "%.1f", sizeInMB)) MB")
        }

        // Test model inference speed
        try testInferenceSpeed(model: model)

        // Analyze optimization opportunities
        analyzeOptimizationOpportunities()
    }

    private func testInferenceSpeed(model: MLModel) throws {
        print("\n🔹 Inference Speed Test:")

        // Create sample input (2 seconds of audio at 16kHz)
        let sampleCount = 32000
        let audioData = (0..<sampleCount).map { _ in Float.random(in: -1.0...1.0) }

        // Create MLMultiArray
        let shape = [1, 1, 32000] as [NSNumber]
        let multiArray = try MLMultiArray(shape: shape, dataType: .float32)

        // Fill with sample data
        for i in 0..<sampleCount {
            multiArray[i] = NSNumber(value: audioData[i])
        }

        // Create input feature
        let inputName = model.modelDescription.inputDescriptionsByName.keys.first!
        let input = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: multiArray)])

        // Time multiple inferences
        let iterations = 10
        let startTime = Date()

        for _ in 0..<iterations {
            let _ = try model.prediction(from: input)
        }

        let endTime = Date()
        let totalTime = endTime.timeIntervalSince(startTime)
        let avgTime = totalTime / Double(iterations)
        let rtf = avgTime / 2.0 // 2 seconds of audio

        print("   Average inference time: \(String(format: "%.3f", avgTime * 1000))ms")
        print("   Real-time factor: \(String(format: "%.2f", rtf))x")
        print("   Status: \(rtf < 1.0 ? "✅ Real-time capable" : "⚠️ Slower than real-time")")
    }

    private func analyzeOptimizationOpportunities() {
        print("\n🎯 MODEL OPTIMIZATION OPPORTUNITIES")
        print(String(repeating: "=", count: 50))

        print("\n1. 🔧 Architecture Improvements:")
        print("   • Current DER: 48.2% (poor overlap detection)")
        print("   • Target DER: 20-25% (competitive performance)")
        print("   • Gap: ~23% DER reduction needed")

        print("\n2. 🎛️ Potential Optimizations:")
        print("   • Model Quantization (FP32 → FP16/INT8)")
        print("   • Feature Engineering (mel-spectrograms, MFCCs)")
        print("   • Temporal Context (sliding windows, RNNs)")
        print("   • Multi-scale Processing (different frame sizes)")
        print("   • Data Augmentation (during training)")
        print("   • Ensemble Methods (multiple models)")

        print("\n3. 🔄 CoreML Specific Optimizations:")
        print("   • Neural Engine optimization")
        print("   • Batch processing improvements")
        print("   • Memory layout optimization")
        print("   • Compute unit selection (CPU/GPU/ANE)")

        print("\n4. 📈 Training Data Improvements:")
        print("   • More diverse overlap scenarios")
        print("   • Longer training sequences")
        print("   • Hard negative mining")
        print("   • Speaker diversity expansion")

        print("\n5. 🎯 Immediate Actions:")
        print("   • Analyze model predictions on AMI corpus")
        print("   • Identify systematic failure patterns")
        print("   • Test different inference configurations")
        print("   • Explore ensemble with multiple models")

        print("\n💡 RECOMMENDATION:")
        print("   Start with inference optimization and feature engineering")
        print("   before retraining the entire model.")
    }

    private func suggestImplementationPlan() {
        print("\n📋 IMPLEMENTATION PLAN")
        print(String(repeating: "=", count: 50))

        print("\nPhase 1 - Model Analysis & Quick Wins (Current)")
        print("   ✅ Analyze current model architecture")
        print("   🔄 Test different inference configurations")
        print("   🔄 Implement feature preprocessing optimizations")
        print("   🔄 Try ensemble approaches")

        print("\nPhase 2 - Advanced Optimizations")
        print("   • Implement temporal smoothing in model")
        print("   • Add multi-scale feature extraction")
        print("   • Optimize for Apple Neural Engine")
        print("   • Test quantization strategies")

        print("\nPhase 3 - Model Retraining (if needed)")
        print("   • Collect additional training data")
        print("   • Implement improved loss functions")
        print("   • Train larger/different architectures")
        print("   • Convert optimized model to CoreML")
    }
}

// MARK: - Main Execution
do {
    let analyzer = ModelAnalyzer()
    try analyzer.analyze()

    print("\n✅ Model analysis completed!")
    print("   🎯 Next: Implement Phase 1 optimizations")

} catch {
    print("❌ Analysis failed: \(error)")
    exit(1)
}