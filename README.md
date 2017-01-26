
# Tree detection in an extract of an orthomosaic

At first, it came to my mind two possible ways of attacking this problem: using traditional image processing from the PIL
library - trying to find pixel patterns withing a certain tolerance margin -  and using computer vision from the
well known OpenCV library. I ended up choosing primarily the OpenCV since it is a widely used library with good documentation and  many
interesting embedded functions like 'template matching' and filters. This code has been my first experience with image processing or
computer vision

## Getting Started

I strongly suggest the reading of the OpenCV [documentation](http://docs.opencv.org/trunk/d6/d00/tutorial_py_root.html)

Simply enter in the console 

```

python pix.py imagename.ext templatename.ext

```
Since one of the methods requires a template, you can simply get it from the image that you want to analyse in the first execution or 
from one of the outputs provided by the filters.

### Prerequisites

```

Python 3.x
NumPy
Matplotlib
OpenCV 3

```

For Windows users, I recommend using the [Anaconda](https://www.continuum.io/downloads) platform. It already comes with Numpy and
Matplotlib. To install OpenCV without headaches simply enter in the command prompt
```

conda install -c menpo opencv3=3.2.0

```

## Template matching

Analysing the example image has proven to be quite tricky; The image doesn't show the clear edge of the trees
and their color aspect varies, making it difficult to find a proper template for comparison. Moreover, the
'template matching' function available in the library has been created to detect only one occurence of
a single template in a given image; trying to detect more than one has returned multiple positives for the 
same tree. Hence, I tried to filter the outputs to prevent this from happening. Special attention to the line in
the **templateMatchingMethod** function:

```Python

loc_filtered = filterMatches(loc, option = 1) #WARNING: option 2 may take much longer to execute

```
The filter is necessary because depending on the image and on the template, many repeating occurences and false positives may be
returned. I suggest you try the three options to find the one that fits the most. Desired precision is an aspect to consider; option 1 is faster than option 2, but the results are much worse. 
More information concerning the filter are provided in the docstring of the **filterMatches** function. 

In the given example, the best result came from the input being the image in greyscale only, with no extra filters aplied but the blur
inside the function.


### Circle matching

It can be observed in the example given that the shape of the trees is quite circular, although their contour is not clear enough.
OpenCV comes with an implementation of the Hough Transform that can detect circles in images. Using it after applying a bilinear filter
to the image has proven to return a good result (not as good as the template matching - option 2 though). This filter reduces the noise
from the background preserving the edges of the objects. To do so, you will eventually have to change some parameters. 
Special attention to:

```python

blur = cv2.bilateralFilter(img,-1,0.02*diag,9)

```

and to

```python

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1,18,param1=1,param2=18,minRadius=10,maxRadius=17)

```

The first is inside the **bilateralFilter** function. More about the parameters can be found [here] (http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#bilateralfilter). For the sigmaColor, 2% of the diagonal has proven to
be good enough. The sigmaSpace will depend on execution time wanted and the hardware. Increasing it can make the program run slower and
the results from using the default value(sigmaSpace=9) has proven to be satisfactory.

The second is inside the **detectCircles** functions. More about the parameters can be found [here](http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghcircles#houghcircles). The param2 is the one that
causes the most noticiable changes. You may take some time to find the ideal values.


## Acknowledgments

* To PixForce for making me explore a topic that was completely unknown to me
* [OpenCV-Python documentation](http://docs.opencv.org/trunk/d6/d00/tutorial_py_root.html)
