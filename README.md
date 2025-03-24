# Fabric Texture Analysis Toolkit
This Python toolkit is designed for advanced fabric texture analysis using statistical methods, enabling the extraction and examination of texture features. Intended for integration into a larger tool, it will be used by students in the _Signal, Image and Video course_ to visually create signal processing workflows. The module emphasizes clear documentation and a user-friendly interface, ensuring seamless integration and ease of use in future educational settings.

## Texture Analysis Techniques
- Autocorrelation Function (ACF)
- Local Binary Patterns (LBP)
- Co-Occurrence Matrices (CoOccur)

## Installation and Usage

1. Clone the repository in your project
  ```bash
  git clone https://github.com/MatteoZanella/siv-texture-analysis.git
  ```
2. Install the requirements
  ```bash
  cd siv-texture-analysis
  pip install -r requirements.txt
  ```
3. Import the modules in your code
```python
from texture.analysis import ACF, CoOccur, LBP
from PIL import Image

# Load an image
image = Image.open('./tests/textures/lena.png')

# Compute Autocorrelation Function
acf = ACF(image)

# Compute Local Binary Patterns
lbp = LBP(image)

# Compute Co-Occurrence Matrices
com = CoOccur(image)
```

## Running Tests
To run the test suite, use Python's unittest module:
```bash
# Run all tests
python -m unittest discover -s tests
```

