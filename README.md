This script analyzes a list of parts or inventory items to find duplicates and similar alternatives based on their text descriptions.

It takes a CSV file as input, generates text embeddings for every item, and calculates the similarity between them. You can choose between a standard TF-IDF approach (good for exact keyword matching) or a Sentence Transformer model (BGE) for semantic matching.

## Dependencies

The script requires Python and the following libraries:

```bash
pip install pandas numpy scikit-learn sentence-transformers
```

## Input Data

The input file should be a CSV using **semicolons (;)** as delimiters.

*   **Required Column:** `DESCRIPTION`
*   **Optional:** `ID`. If `ID` is missing, the script generates one.
*   **Context Columns:** If a description is empty, the script attempts to build one using columns named `Material`, `Size`, `Rating`, `Characteristic`, `Application`, `Code`, `Voltage`, etc.

## Usage

Run the script from the command line:

```bash
python main.py --input parts.csv --outdir ./results --k 5 --embed tfidf
```

### Arguments

*   `--input`: Path to your input CSV file.
*   `--outdir`: Folder where the results will be saved.
*   `--k`: Number of similar matches to find for each part (default is 5).
*   `--embed`: The method used to compare text.
    *   `tfidf`: Faster, matches based on shared words.
    *   `bge`: Slower but smarter, matches based on meaning (uses the `BAAI/bge-large-en-v1.5` model).

## Output

The script produces a "wide" format CSV file. Each row represents an item from your original list, followed by columns showing the top `k` most similar items found in the dataset.

For example, if you set `k=5`, the output columns will look like:
`Original_ID`, `Original_Description`, `Alt1_ID`, `Alt1_Description` ... `Alt5_ID`, `Alt5_Description`.

## Logic

1.  **Data Loading:** Reads the CSV and normalizes the text. Rows with no description and no supporting attributes are logged to a separate file and skipped.
2.  **Vectorization:** Converts the text into numerical vectors using the selected backend (TF-IDF or BGE).
3.  **Similarity:** Calculates the cosine similarity between all vectors.
4.  **Ranking:** For every item, finds the closest neighbors and exports the results.
