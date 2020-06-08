# My humble attempt to implement gesture recognition algorithm with a simple camera

## Requirements

### python 3.7
### Pytorch package [pytorch.org/](https://pytorch.org/)
### OpenCV-Python package [opencv-python-tutroals.readthedocs.io/en/latest/](https://opencv-python-tutroals.readthedocs.io/en/latest/index.html)   
### Pandas package [pandas.pydata.org/](https://pandas.pydata.org/)
### Pygame package [pygame.org/wiki/](https://www.pygame.org/wiki/GettingStarted)

# Creation of the data to be used by the algorithm

## Steps

1. Download the raw data from [kaggle.com](https://www.kaggle.com/gti-upm/leapgestrecog)

2. Preprocess the data.

Start by defining helper functions for the following operations:

  * read the image from a folder.
  * convert the image to gray because color in this case is a noise.
  * Invert the image i.e white becomes black and black white.    

  ```python
  def color_image_gray(folder,filename):
      image = cv2.imread(path.join(folder,filename))
      image2 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
      return ~image2
  ```
  * Convert the image which will be represented as a string of numeric values into binary code so that it can be read by the computer

  ```python
  def convert_image_into_binary_image(img):
      ret,bw_img = cv2.threshold(img,0,255, (cv2.THRESH_BINARY + cv2.THRESH_OTSU))
      return bw_img
  ```
  * Find the contours of the hand i.e fingers and pawn among the rest of the image
  ```python
    def find_countours(image):
        return cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
  ```
3. Define a function which reads all of the images in a folder. For that use two methods from a package in python named ```os``` [docs.python.org/3/library/os.html](https://docs.python.org/3/library/os.html).
The methods are: ```os.path.join``` and ```os.listdir```.

  * Initialize an empty list.
  * use ```listdir``` to put all files from a folder to a list.
  * use ```color_image_gray``` to read the image.
  * resize the image.
  * convert it to binary.
  * reshape it with ```numpy``` [numpy.org/](https://numpy.org/).
  * add it to the first list and repeat.
```python
def read_dataset(folder):
    train_data = []
    for file in listdir(folder):
        img = color_image_gray(folder,file)
        img = cv2.resize(img,(int(160),int(60)))

        img = convert_image_into_binary_image(img)
        img = np.array(img).reshape(9600)
        train_data.append(img)
    train_data = np.array(train_data)
    return train_data
```
4. Do this for all folders in the raw data downloaded in ```1``` and add labels to the images. Practically calling ```read_dataset``` many times and concatenating the resulting arrays. Automate the process as much as possible for personal comfort and convenience.

```python
def create_dataset(folders):
    p = "leapGestRecog/"
    dataset = np.array([])
    for k in range(10):
        folder = folders[k]
        for i in range(10):
            data = read_dataset(p+"0"+str(i)+"/"+folder)
            data = np.insert(data,data.shape[1],[str(i)],axis=1)
            if k == 0 and i == 0:
                dataset = data
            else:
                dataset = np.concatenate((dataset,data))
                print("Step: [{},{}]".format(outher_folder*inner_folder, (k * inner_folder) + i + 1))
    return dataset
```

5. Save the returned database from ```create_dataset``` into the format which suits you best. For me this is ```.csv``` or comma separated values. It will create file which looks like this ```1,2,3,4,5``` and the last element will be the label. Use ```pandas.DataFrame``` and ```pandas.to_csv```.

```python
def dataloaderMain():
    dirs = listdir("leapGestRecog/00")
    folders = sorted(dirs)

    data = create_dataset(folders)
    data = pd.DataFrame(data,index=None)
    data.to_csv('data/database.csv',index=False)
```

6. Call the function ```dataloaderMain()```, be patient.

Now we have a database which we can use to to train a neural network.
