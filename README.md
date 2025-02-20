# DHL2-Task02-Text-Summarization-

## Dataset Overview
**Dataset**: [CNN/Daily Mail Dataset](https://huggingface.co/datasets/cnn_dailymail)  
**Features**:  
- **Article**: Long-form text content (news articles, blogs, etc.)  
- **Highlights**: Human-written summaries of the articles  

**Preprocessing Steps**:
1. **Text Cleaning**:  
   - Removed newline characters (`\n`)  
   - Normalized extra whitespace  
   - Ensured consistent text formatting  

2. **Dataset Splitting**:  
   - Training Data: Used for model training  
   - Validation Data: Used for hyperparameter tuning and evaluation during training  
   - Test Data: Used for final model evaluation  

3. **Tokenization**:  
   - Applied tokenization using the Pegasus tokenizer for abstractive summarization  
   - Used spaCy for sentence splitting in extractive summarization  

4. **Stop Words Removal**:  
   - Removed common stop words (e.g., "the", "is", "and") for extractive summarization  

---

## Models Implemented

### 1. Extractive Summarization (spaCy)
- **Approach**: Frequency-based sentence selection  
- **Key Steps**:  
  - Split text into sentences  
  - Calculate word frequencies (excluding stop words)  
  - Score sentences based on word importance and position  
  - Select top sentences to form the summary  

- **Rationale**:  
  - Lightweight and fast  
  - Preserves factual accuracy by using original sentences  

### 2. Abstractive Summarization (Pegasus)
- **Approach**: Transformer-based sequence-to-sequence model  
- **Key Steps**:  
  - Tokenize input text using Pegasus tokenizer  
  - Generate summary using pre-trained Pegasus model  
  - Fine-tune the model on the CNN/Daily Mail dataset for better performance  

- **Rationale**:  
  - Produces more natural and concise summaries  
  - Capable of paraphrasing and generating novel sentences  

---

## Key Insights

1. **Extractive Summarization**:  
   - Works well for factual content  
   - Preserves original phrasing and accuracy  
   - Limited by the quality of the original sentences  

2. **Abstractive Summarization**:  
   - Generates more human-like summaries  
   - Can handle complex sentence structures  
   - Requires more computational resources  

3. **Fine-Tuning**:  
   - Improves model performance on domain-specific data  
   - Enhances coherence and relevance of generated summaries  

---

## Challenges Faced and Solutions

1. **Challenge**: Long training time for fine-tuning  
   - **Solution**: Used Google Colab GPU and reduced batch size  

2. **Challenge**: Memory limitations  
   - **Solution**: Truncated long articles and used gradient checkpointing  

3. **Challenge**: Dataset size  
   - **Solution**: Worked with a subset of the dataset for initial testing  

4. **Challenge**: Evaluation of summary quality  
   - **Solution**: Used ROUGE scores for quantitative evaluation and manual inspection for qualitative analysis  

---

## How to Run the Code

1. **Setup**:  
   - Install required libraries:  
     ```bash
     !pip install spacy transformers datasets torch sentencepiece
     !python -m spacy download en_core_web_sm
     ```
   - Mount Google Drive:  
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```

2. **Dataset Preparation**:  
   - Unzip the dataset:  
     ```bash
     !unzip "/content/drive/MyDrive/archive (6).zip" -d "/content/data"
     ```

3. **Run the Code**:  
   - Execute the provided Python script in Google Colab.  
   - For fine-tuning, uncomment the `fine_tune_model()` function call.  

4. **Evaluate Results**:  
   - Compare extractive and abstractive summaries.  
   - Test the model on new articles.  

---

## Results

- **Extractive Summarization**:  
  - Produces concise summaries by selecting key sentences.  
  - Example:  
    ```text
    Extractive Summary:
    "The quick brown fox jumps over the lazy dog. The dog barked loudly. The fox ran away."
    ```

- **Abstractive Summarization**:  
  - Generates paraphrased and concise summaries.  
  - Example:  
    ```text
    Abstractive Summary:
    "A fox jumped over a dog, which barked, causing the fox to flee."
    ```

---

## Future Improvements

1. **Model Fine-Tuning**:  
   - Fine-tune on a larger dataset for better performance.  

2. **Evaluation Metrics**:  
   - Implement ROUGE scores for automated evaluation.  

3. **Hybrid Approach**:  
   - Combine extractive and abstractive methods for better results.  

4. **Deployment**:  
   - Deploy the model as a web application for real-world use.  

---

## References

1. [CNN/Daily Mail Dataset](https://huggingface.co/datasets/cnn_dailymail)  
2. [Pegasus Model](https://huggingface.co/google/pegasus-xsum)  
3. [spaCy Documentation](https://spacy.io/)  
4. [Transformers Library](https://huggingface.co/docs/transformers/index)  
