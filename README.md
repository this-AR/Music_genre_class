# Music Genre Classification using Modified CNN

A deep learning implementation for classifying music genres using a Modified Convolutional Neural Network architecture based on MFCC features. This project implements the **Modified CNN** approach from the IEEE research paper "Musical Genre Classification Using Advanced Audio Analysis and Deep Learning Technique."

## 🎵 Project Overview

This system classifies music into **10 genres** using advanced audio preprocessing and deep learning techniques:
- **Genres**: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock
- **Dataset**: GTZAN Music Genre Dataset (1,000 audio files, 30 seconds each)
- **Accuracy**: **85.62%** test accuracy achieved

## 📚 Research Background

This implementation is based on the research paper:
> **"Musical Genre Classification Using Advanced Audio Analysis and Deep Learning Technique"**  
> IEEE Conference Publication  
> DOI: [10.1109/ICAAIC60222.2024.10605044](https://ieeexplore.ieee.org/document/10605044)

**Note**: This project specifically implements the **Modified CNN architecture** from the paper, focusing on enhanced feature extraction through deeper convolutional layers. It does **not** implement the other models (SVM, LSTM, Feedforward NN, etc.) used for comparison in the original study.

## 🏗️ Architecture Overview

### Modified CNN Architecture
The model uses a sequential CNN design with enhanced depth compared to traditional approaches:

```
Input (MFCC Features) → Conv2D Layers → Dense Layers → Output (10 Classes)
```

**Key Components:**
- **4 Convolutional Blocks**: Each with Conv2D → MaxPooling2D → BatchNormalization
- **Filter Progression**: 256 → 128 → 64 → 64 filters
- **Kernel Sizes**: (3,3) for first three layers, (2,2) for the fourth
- **Activation**: ReLU throughout, Softmax for final classification
- **Regularization**: Dropout (0.2) and Batch Normalization for stability
- **Total Parameters**: 423,370 (1.62 MB)

### Architecture Comparison
| Model Type | Conv2D Layers | Filters | Complexity |
|------------|---------------|---------|------------|
| Traditional CNN | 2 | 128, 128 | Lower |
| **Modified CNN** | 4 | 256, 128, 64, 64 | Higher |

The Modified CNN provides richer feature extraction capabilities for better genre classification.

## 🎷 Audio Preprocessing Pipeline

### MFCC Feature Extraction
The system uses **Mel-Frequency Cepstral Coefficients (MFCCs)** which focus on perceptually important audio characteristics:

**Audio Settings:**
- **Sampling Rate**: 22,050 Hz
- **Duration**: 30 seconds per track
- **FFT Size**: 2,048 frames
- **Hop Length**: 512 samples
- **MFCC Components**: 13 coefficients
- **Segmentation**: 10 segments per track (data augmentation)

**Processing Steps:**
1. Load audio files at 22.05 kHz sampling rate
2. Split each 30-second track into 10 segments (66,150 samples each)
3. Extract 13 MFCC features per segment
4. Generate 130 MFCC vectors per segment
5. Reshape for CNN input: `(samples, 13, 130, 1)`

### Why MFCC?
- **Perceptual Relevance**: Mimics human auditory system
- **Noise Robustness**: Less affected by background noise
- **Computational Efficiency**: Compressed representation
- **Genre Discrimination**: Captures unique sound characteristics

## 🚀 Training Configuration

### Model Training
- **Optimizer**: Adam (learning_rate=0.0001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Train/Test Split**: 70/30 (6,930 train, 2,970 test)

### Advanced Training Strategies
- **ModelCheckpoint**: Saves best model based on validation accuracy
- **EarlyStopping**: Prevents overfitting (patience=10)
- **ReduceLROnPlateau**: Dynamic learning rate reduction (factor=0.5, patience=5)
- **Batch Normalization**: Improves training stability
- **Dropout**: Prevents overfitting (rate=0.2)

## 📊 Results & Performance

### Final Results
- **Test Accuracy**: **85.62%**
- **Best Validation Accuracy**: 85.62% (Epoch 49)
- **Training Samples**: 6,930
- **Test Samples**: 2,970
- **Total Segments Processed**: 9,900

### Training Progress
- **Early Performance**: 42.73% validation accuracy (Epoch 1)
- **Steady Improvement**: Reached 80%+ by Epoch 18
- **Final Convergence**: 85.62% at Epoch 49
- **Training Accuracy**: 99.37% (final epoch)

### Genre Classes
```python
['country', 'jazz', 'hiphop', 'blues', 'rock', 
 'classical', 'metal', 'disco', 'pop', 'reggae']
```

## 🛠️ Installation & Usage

### Prerequisites
```bash
# Core dependencies
pip install tensorflow>=2.0
pip install librosa
pip install numpy
pip install scikit-learn
pip install kagglehub
pip install tqdm
pip install matplotlib
```

### Quick Start

#### Option 1: Google Colab (Recommended)
1. Open the notebook in Google Colab
2. Run all cells - the dataset downloads automatically
3. GPU acceleration is enabled by default

#### Option 2: Local Setup
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Kaggle Setup** (for dataset download):
   - Create `kaggle.json` with your Kaggle API credentials
   - Place in `~/.kaggle/` directory

3. **Run the script**:
   ```bash
   python mcnn_song_classifier.py
   ```

### Dataset
The GTZAN dataset is automatically downloaded using KaggleHub:
- **Source**: `andradaolteanu/gtzan-dataset-music-genre-classification`
- **Size**: 1.21 GB
- **Format**: WAV files, 30 seconds each
- **Structure**: 10 genres × 100 files each

### GPU Support
- **Automatic Detection**: Code automatically detects and uses GPU if available
- **Memory Growth**: Dynamic GPU memory allocation
- **CPU Fallback**: Works on CPU if GPU unavailable

## 📈 Model Outputs

### Generated Files
- **Model**: `modified_cnn_gtzan_10_classes.h5`
- **Dataset**: `data_gtzan.json` (preprocessed MFCC features)
- **Training Plot**: `training_history.png` (accuracy/loss curves)

### Inference Example
```python
# Load trained model
from tensorflow.keras.models import load_model
model = load_model('modified_cnn_gtzan_10_classes.h5')

# Predict genre for new audio file
prediction = model.predict(mfcc_features)
genre = genres[np.argmax(prediction)]
```

## 🔬 Technical Highlights

### Key Innovations from the Paper
1. **Enhanced Feature Extraction**: Modified CNN with 4 convolutional layers vs. 2 in traditional approaches
2. **Progressive Filter Reduction**: 256→128→64→64 filter progression for hierarchical feature learning
3. **Batch Normalization**: Improves training stability and convergence
4. **Audio Segmentation**: 10 segments per track for data augmentation

### Performance Optimizations
- **Efficient Data Loading**: JSON-based preprocessed features
- **Memory Management**: Dynamic GPU memory growth
- **Batch Processing**: Optimized batch size for training efficiency
- **Early Stopping**: Prevents overfitting and reduces training time

## 🚨 Limitations & Scope

### Current Implementation
- ✅ **Modified CNN Architecture**: Fully implemented
- ❌ **Other Classifiers from Paper**: Not implemented (e.g., SVM, LSTM, kNN)
- ❌ **Traditional CNN Comparison in Code**: Not included
- ❌ **Spectrogram Features**: Only MFCC features used
- ❌ **Real-time Classification**: Batch processing only

### Dataset Limitations
- **Limited Diversity**: 100 samples per genre may not capture full variety
- **30-Second Clips**: May miss longer musical patterns
- **Audio Quality**: Some files may have quality issues
- **Genre Boundaries**: Some songs may span multiple genres

> ⚠️ _This project focuses solely on the modified CNN model proposed in the paper "Musical Genre Classification Using Advanced Audio Analysis and Deep Learning Technique." Models like SVM, Feedforward NN, and RNN-LSTM shown in the research comparison table were **not** implemented in this version._

## 🔮 Future Enhancements

### Potential Improvements
1. **Other Architectures**: Implement and benchmark CNN, SVM, LSTM, etc. from the paper
2. **Multi-Modal Features**: Combine MFCC with spectrograms
3. **Data Augmentation**: Time-stretching, pitch-shifting
4. **Transfer Learning**: Pre-trained audio models
5. **Real-time Processing**: Streaming audio classification
6. **Genre Fusion**: Handle multi-genre classifications

## 📜 Citation

### Original Paper
```bibtex
@inproceedings{musical_genre_classification_2024,
  title={Musical Genre Classification Using Advanced Audio Analysis and Deep Learning Technique},
  author={[Author Names]},
  booktitle={IEEE Conference},
  year={2024},
  doi={10.1109/ICAAIC60222.2024.10605044},
  url={https://ieeexplore.ieee.org/document/10605044}
}
```

### This Implementation
```bibtex
@misc{modified_cnn_gtzan_2024,
  title={Music Genre Classification using Modified CNN - GTZAN Implementation},
  author={[Your Name]},
  year={2024},
  url={[Your Repository URL]}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**🎵 Happy Genre Classification!** 🎵
