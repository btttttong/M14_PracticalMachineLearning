# Cat Breed Search with CLIP

This project demonstrates how to use a CLIP model to search for cat breeds based on textual descriptions.

## Setup

1. Clone this repository to your local machine.

2. Navigate to the project directory.

3. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

5. Install the official CLIP package from OpenAI's GitHub:
    ```bash
    pip install git+https://github.com/openai/CLIP.git
    ```

6. Download the CSV file:
    ```bash
    python download_csv.py
    ```

7. Ensure you have the necessary images in the `/data/images/breeds` directory.

8. Create the FAISS index:
    ```bash
    python create_faiss_index.py
    ```

9. Run the Streamlit app:
    ```bash
    streamlit run main.py
    ```

10. Enter a description of a cat breed in the input box and click "Find Image" to see the top matching breeds.


