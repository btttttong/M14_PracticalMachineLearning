# Cat Breed Image Finder

This project is a Streamlit application that uses the CLIP model and FAISS index to find images of cat breeds based on text input.

## Prerequisites

- Python 3.8+
- Virtual environment (recommended)

## Setup

1. **Clone the repository** (if this project is in a Git repository):

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create and activate a virtual environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Download and prepare the dataset**:

    Run the `download_csv.py` script to download the dataset and prepare the CSV file.

    ```bash
    python download_csv.py
    ```

5. **Create the FAISS index**:

    Run the `create_faiss_index.py` script to create the FAISS index from the dataset.

    ```bash
    python create_faiss_index.py
    ```

## Running the Application

1. **Run the Streamlit application**:

    ```bash
    streamlit run main.py
    ```

2. **Open your web browser** and go to `http://localhost:8501` to use the application.

## Project Structure

/project_directory
    /data
        faiss.index
        train_cat_breeds.csv
        /images
    /model
        model.py
    main.py
    create_faiss_index.py
    download_csv.py
    README.md
    requirements.txt
    verify_imports.py


## Notes

- The `main.py` script sets the `KMP_DUPLICATE_LIB_OK` environment variable to avoid OpenMP runtime conflicts.
- Ensure that the `model.py` file is correctly referenced in the `main.py` script.


## Troubleshooting

- **OpenMP Runtime Conflict**: The `main.py` script includes a workaround by setting `KMP_DUPLICATE_LIB_OK=TRUE`.
- **Dependency Issues**: Ensure you are using the correct versions of dependencies as specified in `requirements.txt`.

## Acknowledgements

- This project uses the [CLIP model](https://github.com/openai/CLIP) by OpenAI.