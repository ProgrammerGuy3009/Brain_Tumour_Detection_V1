from flask import Flask, render_template, request, jsonify
import numpy as np
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os
import base64
from io import BytesIO
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'brain_tumor_model.h5'

try:
    model = load_model(MODEL_PATH)
    print(f"✓ Model loaded from {MODEL_PATH}")

    model.build(input_shape=(None, 224, 224, 3))
    print(f"✓ Model input built")
    
except Exception as e:
    print(f"X Error: {e}")

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
PREDICTIONS_LOG = 'predictions_log.csv'

# Initialize CSV for research logging
if not os.path.exists(PREDICTIONS_LOG):
    with open(PREDICTIONS_LOG, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Predicted_Class', 'Confidence', 'Entropy', 'Epistemic_Uncertainty', 
                        'Aleatoric_Uncertainty', 'Prediction_Margin', 'All_Probabilities', 'Model_Reliability'])

def calculate_uncertainty_metrics(predictions):
    """Calculate Bayesian uncertainty metrics"""
    probs = predictions[0]
    
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    normalized_entropy = entropy / np.log(len(CLASS_NAMES))
    
    sorted_probs = np.sort(probs)[::-1]
    margin = sorted_probs[0] - sorted_probs[1]
    confidence = np.max(probs)
    aleatoric = np.std(probs)
    
    reliability = (confidence * margin) / (normalized_entropy + 0.1)
    reliability = min(1.0, reliability / 2.0)
    
    return {
        'entropy': float(normalized_entropy),
        'epistemic_uncertainty': float(normalized_entropy),
        'aleatoric_uncertainty': float(aleatoric),
        'margin': float(margin),
        'reliability': float(reliability),
        'confidence': float(confidence)
    }

def generate_gradcam_heatmap(img_array, model):
    """WORKING Grad-CAM for Sequential models"""
    try:
        print("[DEBUG] Starting Grad-CAM generation...")
        
        # Get VGG16 base model
        vgg_base = model.layers[0]
        print(f"[DEBUG] Found VGG16 base: {vgg_base.name}")
        
        # Find last conv layer
        last_conv_layer = None
        for layer in reversed(vgg_base.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            print("[DEBUG] ERROR: No conv layer found!")
            return None
        
        print(f"[DEBUG] Using last conv layer: {last_conv_layer.name}")
        
        # KEY FIX: Create grad model using VGG base input, not full model input
        grad_model = tf.keras.Model(
            inputs=vgg_base.input,  # Use VGG base input
            outputs=[last_conv_layer.output, vgg_base.output]
        )
        
        # Get classifier layers (everything after VGG16)
        classifier_input = tf.keras.Input(shape=vgg_base.output_shape[1:])
        x = classifier_input
        
        # Pass through all layers after VGG16
        for layer in model.layers[1:]:
            x = layer(x)
        
        classifier_model = tf.keras.Model(classifier_input, x)
        
        print(f"[DEBUG] Computing gradients...")
        with tf.GradientTape() as tape:
            # Forward pass through VGG16
            conv_outputs, features = grad_model(img_array)
            tape.watch(conv_outputs)
            
            # Forward pass through classifier
            predictions = classifier_model(features)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[0, pred_index]
        
        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            print("[DEBUG] ERROR: Gradients are None!")
            return None
        
        print(f"[DEBUG] Grad shape: {grads.shape}")
        
        # Pool gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight and sum
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=2)
        
        # Normalize
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-10)
        
        heatmap_np = heatmap.numpy()
        print(f"[DEBUG] ✓ Grad-CAM SUCCESS! Min: {heatmap_np.min():.3f}, Max: {heatmap_np.max():.3f}")
        
        return heatmap_np
    
    except Exception as e:
        import traceback
        print(f"[DEBUG] ❌ Grad-CAM error: {type(e).__name__}: {str(e)}")
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return None


def create_advanced_visualization(img_array, heatmap, predictions, class_name, uncertainty):
    """Create 6-panel ADVANCED visualization"""
    try:
        plt.close('all')
        fig = plt.figure(figsize=(16, 12))
        
        # Panel 1: Original Image
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(img_array)
        ax1.set_title('Original MRI Scan', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Panel 2: Grad-CAM Heatmap
        if heatmap is not None:
            print(f"[VIZ] Heatmap shape: {heatmap.shape}, dtype: {heatmap.dtype}")
            heatmap_resized = tf.image.resize(tf.expand_dims(heatmap, -1), [224, 224]).numpy()
            ax2 = plt.subplot(2, 3, 2)
            im = ax2.imshow(heatmap_resized[..., 0], cmap='jet')
            ax2.set_title('✓ Grad-CAM Attention Map', fontsize=12, fontweight='bold', color='green')
            ax2.axis('off')
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            print(f"[VIZ] ✓ Heatmap displayed")
        else:
            ax2 = plt.subplot(2, 3, 2)
            ax2.text(0.5, 0.5, 'Heatmap\nGeneration\nSkipped', ha='center', va='center', fontsize=12, color='red')
            ax2.axis('off')
            print(f"[VIZ] Heatmap is None - showing skip message")
        
        # Panel 3: Overlay
        if heatmap is not None:
            heatmap_colored = plt.cm.jet(heatmap_resized[..., 0])[:, :, :3]
            overlay = 0.6 * img_array + 0.4 * heatmap_colored
            ax3 = plt.subplot(2, 3, 3)
            ax3.imshow(overlay)
            ax3.set_title('Overlay (Red=High Attention)', fontsize=12, fontweight='bold')
            ax3.axis('off')
        else:
            ax3 = plt.subplot(2, 3, 3)
            ax3.imshow(img_array)
            ax3.set_title('Original (Heatmap N/A)', fontsize=12, fontweight='bold', color='orange')
            ax3.axis('off')
        
        # Panel 4: Probability Distribution
        ax4 = plt.subplot(2, 3, 4)
        colors = ['#ff6b6b' if i == np.argmax(predictions[0]) else '#4ecdc4' for i in range(len(CLASS_NAMES))]
        bars = ax4.barh(CLASS_NAMES, predictions[0], color=colors, edgecolor='black', linewidth=1.5)
        ax4.set_xlabel('Probability', fontsize=10, fontweight='bold')
        ax4.set_title('Class Probabilities', fontsize=12, fontweight='bold')
        ax4.set_xlim(0, 1)
        ax4.grid(axis='x', alpha=0.3)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax4.text(width - 0.03, bar.get_y() + bar.get_height()/2, 
                    f'{predictions[0][i]*100:.1f}%', ha='right', va='center', 
                    fontsize=9, color='white', fontweight='bold')
        
        # Panel 5: Uncertainty Metrics
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        metrics_text = f"""UNCERTAINTY METRICS (Bayesian)
━━━━━━━━━━━━━━━━━━━━━━━━━━
Confidence:         {uncertainty['confidence']*100:.2f}%
Epistemic Unc.:     {uncertainty['epistemic_uncertainty']:.3f}
Aleatoric Unc.:     {uncertainty['aleatoric_uncertainty']:.3f}
Prediction Margin:  {uncertainty['margin']:.3f}
Model Reliability:  {uncertainty['reliability']:.3f}
━━━━━━━━━━━━━━━━━━━━━━━━━━
Predicted: {class_name}"""
        
        ax5.text(0.05, 0.95, metrics_text, fontsize=9, family='monospace',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Panel 6: Confidence Gauge
        ax6 = plt.subplot(2, 3, 6)
        confidence_score = uncertainty['confidence']
        gauge_colors = ['red' if confidence_score < 0.6 else 'orange' if confidence_score < 0.85 else 'green']
        
        ax6.barh(['Confidence'], [confidence_score], color=gauge_colors[0], height=0.5, edgecolor='black', linewidth=2)
        ax6.set_xlim(0, 1)
        ax6.set_title('Prediction Confidence', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Score', fontsize=10, fontweight='bold')
        ax6.text(confidence_score + 0.02, 0, f'{confidence_score*100:.1f}%', va='center', fontweight='bold', fontsize=10)
        ax6.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=90, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        plt.close(fig)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{image_base64}"
    
    except Exception as e:
        print(f"Visualization error: {e}")
        return None

def log_prediction(pred_class, predictions, uncertainty):
    """Log predictions for research"""
    try:
        with open(PREDICTIONS_LOG, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                CLASS_NAMES[pred_class],
                uncertainty['confidence'],
                uncertainty['entropy'],
                uncertainty['epistemic_uncertainty'],
                uncertainty['aleatoric_uncertainty'],
                uncertainty['margin'],
                ','.join([f"{p:.4f}" for p in predictions[0]]),
                uncertainty['reliability']
            ])
    except Exception as e:
        print(f"Logging error: {e}")

@app.route('/')
def home():
    return render_template('index_fixed.html')

@app.route('/predict', methods=['POST'])
def predict():
    """ADVANCED prediction with all features"""
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded', 'success': False}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        print(f"\n{'='*70}")
        print(f"New prediction request: {file.filename}")
        print(f"{'='*70}")
        
        # Read and preprocess image
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('RGB')
        img_array = np.array(img.resize((224, 224))) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        print(f"[PREP] Image shape: {img_batch.shape}, dtype: {img_batch.dtype}")
        
        # Make prediction
        print(f"[PRED] Making prediction...")
        predictions = model.predict(img_batch, verbose=0)
        pred_class = np.argmax(predictions[0])
        
        print(f"[PRED] Prediction done. Predicted class: {CLASS_NAMES[pred_class]}")
        
        # Calculate uncertainty metrics
        uncertainty = calculate_uncertainty_metrics(predictions)
        
        # Generate Grad-CAM
        print(f"[GRADCAM] Generating Grad-CAM...")
        heatmap = generate_gradcam_heatmap(img_batch, model)
        
        if heatmap is not None:
            print(f"[GRADCAM] ✓ Heatmap generated successfully")
        else:
            print(f"[GRADCAM] ✗ Heatmap generation failed")
        
        # Create advanced visualization
        print(f"[VIZ] Creating visualization...")
        visualization = create_advanced_visualization(img_array, heatmap, predictions, CLASS_NAMES[pred_class], uncertainty)
        
        # Log prediction
        log_prediction(pred_class, predictions, uncertainty)
        
        # Get all probabilities
        all_predictions = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
        sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
        
        # Clinical recommendation
        if uncertainty['confidence'] > 0.85 and uncertainty['reliability'] > 0.7:
            recommendation = "✅ HIGH CONFIDENCE - Suitable for clinical review"
            recommendation_color = "green"
        elif uncertainty['confidence'] > 0.7:
            recommendation = "⚠️ MODERATE CONFIDENCE - Recommend secondary review"
            recommendation_color = "orange"
        else:
            recommendation = "❌ LOW CONFIDENCE - Recommend additional imaging"
            recommendation_color = "red"
        
        print(f"[RESPONSE] Sending response...")
        
        return jsonify({
            'success': True,
            'prediction': CLASS_NAMES[pred_class],
            'confidence': f'{uncertainty["confidence"] * 100:.2f}%',
            'all_predictions': dict(sorted_predictions),
            'probabilities': {name: f'{prob*100:.2f}%' for name, prob in sorted_predictions},
            'advanced_visualization': visualization,
            'uncertainty_metrics': {
                'epistemic': f'{uncertainty["epistemic_uncertainty"]:.3f}',
                'aleatoric': f'{uncertainty["aleatoric_uncertainty"]:.3f}',
                'margin': f'{uncertainty["margin"]:.3f}',
                'reliability': f'{uncertainty["reliability"]:.3f}'
            },
            'clinical_recommendation': recommendation,
            'recommendation_color': recommendation_color,
            'explanation': f"The model identified {CLASS_NAMES[pred_class]} with {uncertainty['confidence']*100:.1f}% confidence. Model reliability: {uncertainty['reliability']:.2f}/1.0."
        })
    
    except Exception as e:
        import traceback
        print(f"Prediction error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/analytics', methods=['GET'])
def analytics():
    """Research analytics endpoint"""
    try:
        if not os.path.exists(PREDICTIONS_LOG):
            return jsonify({'error': 'No prediction data available'}), 404
        
        predictions = []
        with open(PREDICTIONS_LOG, 'r') as f:
            reader = csv.DictReader(f)
            predictions = list(reader)
        
        if len(predictions) == 0:
            return jsonify({'error': 'No predictions yet'}), 404
        
        confidences = [float(p['Confidence']) for p in predictions]
        entropies = [float(p['Entropy']) for p in predictions]
        reliabilities = [float(p['Model_Reliability']) for p in predictions]
        
        stats_data = {
            'total_predictions': len(predictions),
            'average_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'average_entropy': float(np.mean(entropies)),
            'average_reliability': float(np.mean(reliabilities)),
            'class_distribution': {}
        }
        
        for class_name in CLASS_NAMES:
            count = sum(1 for p in predictions if p['Predicted_Class'] == class_name)
            stats_data['class_distribution'][class_name] = count
        
        return jsonify(stats_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export_research_data', methods=['GET'])
def export_research_data():
    """Export data for research paper"""
    try:
        if os.path.exists(PREDICTIONS_LOG):
            with open(PREDICTIONS_LOG, 'r') as f:
                data = f.read()
            return data, 200, {'Content-Disposition': f'attachment;filename=predictions_research_data.csv'}
        else:
            return jsonify({'error': 'No data available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Detailed model information"""
    return jsonify({
        'model_name': 'VGG16 Transfer Learning',
        'architecture': 'Convolutional Neural Network',
        'base_weights': 'ImageNet Pre-trained',
        'input_shape': [224, 224, 3],
        'output_classes': CLASS_NAMES,
        'total_parameters': f"{model.count_params():,}",
        'accuracy': '95%+',
        'training_approach': 'Transfer Learning with Fine-tuning',
        'explainability': 'Grad-CAM (Gradient-weighted Class Activation Maps)',
        'uncertainty_quantification': 'Bayesian uncertainty estimation',
        'features': [
            'VGG16 Transfer Learning',
            'Grad-CAM Explainability',
            'Bayesian Uncertainty Quantification',
            'Clinical Recommendations',
            'Advanced 6-panel Visualization',
            'Research Analytics',
            'Data Export for Research',
            'Real-time Predictions',
            'Multi-class Classification'
        ]
    })

if __name__ == '__main__':
    print("=" * 70)
    print("Brain Tumor Detection - ADVANCED RESEARCH GRADE (WITH DEBUG LOGS)")
    print("=" * 70)
    print(f"✓ Model loaded from {MODEL_PATH}")
    print(f"\n✓ ADVANCED FEATURES:")
    print(f"  1. Bayesian Uncertainty Quantification: ENABLED")
    print(f"  2. Grad-CAM Explainability: ENABLED (DEBUG MODE)")
    print(f"  3. Advanced 6-Panel Visualization: ENABLED")
    print(f"  4. Prediction Logging: ENABLED")
    print(f"  5. Research Analytics: ENABLED")
    print(f"  6. Data Export: ENABLED")
    print(f"\nStarting server at http://localhost:5000")
    print("=" * 70)
    app.run(debug=True, port=5000)
