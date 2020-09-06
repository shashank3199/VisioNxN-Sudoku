![Version Tag](https://img.shields.io/badge/Version-1.0.0-blue.svg)
![Python Tag](https://img.shields.io/badge/Python-3-green.svg)
![OpenCV Tag](https://img.shields.io/badge/OpenCV-4.2.0-yellow.svg)
![PyTorch Tag](https://img.shields.io/badge/PyTorch-1.6.0+cpu-orange.svg)
![PyGame Tag](https://img.shields.io/badge/PyGame-1.9.6-blueviolet.svg)


# <img width="64" height="64" src="./.images/game_logo.png"> &nbsp; VisioNxN Sudoku - 

This project aims to solve any sudoku of N dimension, where N is a non-prime. The project uses pygame for creating the Graphical User Interface. The project is implemented in two forms -

- An option to load an image saved on the system or use the webcam to feed image to the program and then play the game on the system.
- An option to use <b>Augmented Reality</b> and solve the sudoku by showing the puzzle to the webcam.

The project uses <i>[Dancing Links](https://en.wikipedia.org/wiki/Dancing_Links)</i> in the form of <i>[Algorithm X](https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X)</i> to find the solution of the Sudoku puzzle. Sudoku is a well known NP-Complete problem and Algorithm X is a means of implementing a form of greedy depth first search to find the appropriate solution. For more understanding on the Sudoku Algorithm, read [here](./Sudoku/README.md). For understanding the Image Processing approach, read [here](./Image_Processing/README.md).

## Game Images - 

<p align="center">

<img width="480" height="480" src="./.images/Demo/1.png">
<br>Opening Screen<br><br>

<img width="480" height="480" src="./.images/Demo/2.png">
<br>Play Game<br><br>

<img width="480" height="480" src="./.images/Demo/3.png">
<br>Sample Game Loaded<br><br>

<img width="480" height="480" src="./.images/Demo/4.png">
<br>Load from file<br><br>

<img width="480" height="480" src="./.images/Demo/5.png">
<br>Select File<br><br>

<img width="480" height="480" src="./.images/Demo/6.png">
<br>Succesfully Loaded<br><br>

<img width="480" height="480" src="./.images/Demo/7.png">
<br>Load from Camera: 9 X 9<br><br>

<img width="480" height="480" src="./.images/Demo/8.png">
<br>Succesfully Loaded<br><br>

<img width="480" height="480" src="./.images/Demo/9.png">
<br>Load from Camera: 8 X 8<br><br>

<img width="480" height="480" src="./.images/Demo/10.png">
<br>Succesfully Loaded<br><br>

<img width="480" height="480" src="./.images/Demo/11.png">
<br>Load from Camera: 6 X 6<br><br>

<img width="480" height="480" src="./.images/Demo/12.png">
<br>Succesfully Loaded<br><br>

<img width="480" height="480" src="./.images/Demo/13.png">
<br>Load from Camera: 4 X 4<br><br>

<img width="480" height="480" src="./.images/Demo/14.png">
<br>Succesfully Loaded<br><br>

<img width="480" height="480" src="./.images/Demo/15.png">
<br>When an invalid image is clicked<br><br>

<img width="480" height="480" src="./.images/Demo/16.png">
<br>Start Playng the Game<br><br>

<img width="480" height="480" src="./.images/Demo/17.png">
<br>Solving using the Solve Button<br><br>

<img width="480" height="480" src="./.images/Demo/18.png">
<br>Game Won<br><br>

<img width="480" height="480" src="./.images/Demo/19.png">
<br>Augmented Reality Option<br><br>

<img width="560" height="480" src="./.images/Demo/20.png">
<br>Augmented Reality Test: 4 X 4<br><br>

<img width="560" height="480" src="./.images/Demo/21.png">
<br>Augmented Reality Test: 6 X 6<br><br>

<img width="560" height="480" src="./.images/Demo/22.png">
<br>Augmented Reality Test: 8 X 8<br><br>

<img width="560" height="480" src="./.images/Demo/23.png">
<br>Augmented Reality Test: 9 X 9<br><br>

<img width="560" height="480" src="./.images/Demo/24.png">
<br>Augmented Reality Test: 4 X 4<br><br>

<img width="560" height="480" src="./.images/Demo/25.png">
<br>Augmented Reality Test: 6 X 6<br><br>

<img width="560" height="480" src="./.images/Demo/26.png">
<br>Augmented Reality Test: 8 X 8<br><br>

<img width="560" height="480" src="./.images/Demo/27.png">
<br>Augmented Reality Test: 9 X 9<br><br>

</p>

### Files in the Repository - 
The files in the repository are :

#### GUI -

-   ##### \_\_init__.py
    The \_\_init__.py file is to make Python treat directories containing the file as packages.

-   ##### button.py
    This file contains the class for implementing a pygame button.

-   ##### camera_windows.py
	This file contains the class for implementing a camera screen in pygame.

#### Image_Processing -

-   ##### data
    This directory contains the images of digits used as the training dataset for the image classifier.
    It contains images pertaining to digits of the char74k dataset.

-	##### char74k-cnn.pth
	It is the file containing the weights of the trained model on the `data` directory.

-	##### char74k_dataset.tgz
	This zip file contains the complete char74k dataset.

-   ##### classifier.py
    This file contains the CNN Classifier for digit recognition.

-   ##### \_\_init__.py
    The \_\_init__.py file is to make Python treat directories containing the file as packages.

-   ##### process_image.py
	This file contains the class for recognition of the sudoku from image.

-   ##### pytorch_gpu_assist.py
	This file contains helper functions to implement training on GPUs.

-   ##### README.md
    This file contains details about the algorithm and its implementation.

#### Samples
This directory contains sample images of sudoku puzzles that can be used as test cases for loading the image from file option in the program.

#### Sudoku -

-   ##### DancingLinks.pdf
    This is a copy of the original paper written by `Donald Knuth` on the concept of Dancing Links.

-   ##### \_\_init__.py
    The \_\_init__.py file is to make Python treat directories containing the file as packages.

-	##### README.md
    This file contains details about the algorithm and its implementation.

-   ##### sudoku.py
	This file contains class to solve Sudoku puzzle.

#### .images
This directory contains the images for the icons and other media for the README File.

#### cli_main.py
This file can be used to solve sudoku of any dimension using CLI. To run the program -

```bash
python3 cli_main.py
```

#### game_window.py
This file contains the class for implementing the GUI for the program.

#### \_\_init__.py
The \_\_init__.py file is to make Python treat directories containing the file as packages.

#### main.py
This file is used as the driver code to start the program. To start the program  -

```bash
python3 main.py
```

#### README.md
The Description file containing details about the repository. The file that you looking at right now.

#### requirements.txt
This file contains the respective packages needed to be installed. To install the respective packages, use -

```bash
pip3 install -r requirements.txt

or 

pip install -r requirements.txt
```  

#### sample.npy
This file contains the default sudoku puzzle which will get loaded incase not last loaded file is found.

<b>Fun Fact</b>: <i>This specific puzzle is designed to work against backtracking as a solution for sudoku and will take almost forever to solve it using backtracking.</i>




## Bibliography

- <b>Game Icon:</b> Icon made by [Freepik](https://www.flaticon.com/authors/freepik) from [flaticons.com](https://www.flaticon.com/).
- <b>Camera Icon:</b> Icon made by [Freepik](https://www.flaticon.com/free-icon/camera_2088898) from [flaticons.com](https://www.flaticon.com/).
- <b>Home Icon:</b> Icon made by [Freepik](https://www.flaticon.com/authors/freepik) from [flaticons.com](https://www.flaticon.com/).
- <b>Rounded Rectangle for Buttons:</b> The code for making rounded rectangle surface in pygame is adapted from [here](https://www.pygame.org/project-AAfilledRoundedRect-2349-.html).
- <b>Char74k Dataset:</b> The dataset used for training the CNN to recognize the digits can be found [here](
http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/).
- <b>CNN Model:</b> The Model Architecture is adapted from [here](https://www.kaggle.com/juiyangchang/cnn-with-pytorch-0-995-accuracy).
- <b>Sudoku Solver:</b> The code for implementing Sudoku as an exact cover problem is adapted from [here](https://www.cs.mcgill.ca/~aassaf9/python/algorithm_x.html). 

[![Developers Tag]( https://img.shields.io/badge/Developer-shashank3199-red.svg )]( https://github.com/shashank3199 )<br>