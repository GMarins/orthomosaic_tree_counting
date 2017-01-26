
## Notes regarding performance and improvements
Since python is an interpreted language, the same algorithms implemented in the OpenCV library for C/C++ would already be considerably
faster.

Many OpenCV functions are already optimized using some features. More about the OpenCV optimization can be found 
[here](http://docs.opencv.org/trunk/dc/d71/tutorial_py_optimization.html). 

Even though I did not have access to what happens behind the OpenCV funcionts, I have tried to avoid using features that would increase
the execution time. I have avoided using *NumPy* functions where there where alternatives in the Pythons' *math* library. As stated in
the link above, *NumPy* operations may take 20x to 100x longer to execute that Python native scripts.

Although the script can run with a good perfomance when the desired result does not require full precision, the result can be quite
different when required. The *option = 2* in **filterMatches** function is O(n²), making the execution increasingly slower the bigger the
input image is. Since we have an array of (x,y) tuples and close points may be in distant positions in the array (even if sorted in x or y) , is difficult to find an intuitive alternative to nested loops. However, the execution time of *option = 2* can be acceptable provided
that the filter is well configured: having the tuples sorted *([(x1,y1),(x2,y2),...], y1 < y2*; as in the normal output of the OpenCV function) 
and checking each position with respect to N positions forward, being N a percentage of the size of the array where it is the most probable range for a repeted occurence to happen.

For what I could conclude from my study, is that the best way to improve would be to use another feature of OpenCV: the usage of 
Haar Cascades ( [Example](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html)). OpenCV already comes 
with some cascades for face recognition. For detecting the trees in the given example, it would be necessary to create a cascade from zero
using machine learning algorithms as proposed [here](http://johnallen.github.io/opencv-object-detection-tutorial/). Although it may be
costly (it would probably require some hours to create the cascade in a server), it is a more trustable approach than circle and template
matching since it trains the machine with  thousands of "positives" and  "negatives" of the results desired and it is fast and scalable.
