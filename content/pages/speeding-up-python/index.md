---
date: '2024-12-10T11:44:51-05:00'
draft: false
title: 'Speeding Up Python'
---


# Introduction 
In a recent project I ran into an issue where i was creating a ros2 node, which converts images from a camera topic converted depth image to a Point Cloud. The ML library the model is was built on is tinygrad which forces the node implementation to be in python. This forces the use of open3d python bindings for the second part of the pipeline converting the depth image to the pointcloud This normally is quite simple to do using a third party library called open3d. I had run into a lack of this dependency as in the converion from python 3.11 to 3.12 is not as straight forward for the library so it has not been realeased for python 3.12.  

Implementing to convert from image coordinates to world coordinates for a depth image is really simple and is goverened by the following equtions


$$ z = d / scaling $$


$$ x = (u - c_x) * z/fx $$

$$ y = (v - c_y) * z/fy $$


```python
def convertToPointCloud(arr:np.ndarray,cx:float,cy:float,fx:float,fy:float)->np.ndarray:
  height:int
  width:int
  depth_scale:float = 1000
  height, width = arr.shape
  def indexConversion(inp, width=width,height=height):
    return (inp%width,inp//width)
  pcl_list = []
  for d in range(height*width):
    u,v = indexConversion(d)
    z = -arr[v,u]/depth_scale
    x = ((u - cx) * z)/fx
    y = ((v - cy) * z)/fy
    pcl_list.append([x, y, z])
  return np.array(pcl_list,dtype=np.float32)
```
The above is a non vectorized implementation to convert the depth image to point cloud. This was the first implementation and when implemented I noticed a drastic slowdown in the amount of frames processed from about 70+ fps at 320,240 and we were now getting soureces of latency at around 14 fps. This is rather problematic for a system that needs to run in real-time.  to improve this we can use cython a process of compiling python to code C enabling speed ups while keeping python style.  This does introduce a build step to build the code.

```python
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  ext_modules=cythonize("file_with_method.pyx",compiler_directives={"language_level":3}),
    include_dirs=[numpy.get_include()] # required to be able to use numpy arrays
)
```
Cython is really neat as it allows you to type your python and you can compile it and easily cut processing time without actually changing the code.  Then you can write it the cython way which is different a hybrid syntax that is rather clike in compared to python but runs more quickly with a little extra time. Doping this we converted using both different stylesbot of which look lke the following.  Since the code is in c your able to translate over multithreading without the gil getting in the way making it easier to write multi threaded code. Their are some drawback however for really small tasks you maybe better off not using cython as their is overhead in translating the python object to a something usable in c code. If Cython is a great way to speed up numpy code as it reduces allows the code to touch python less and operate in compile land.  

```python
import numpy as np
cimport numpy as cnp
import cython 
cnp.import_array() # allows usage for numpy typing

DTYPE = np.float32
ctypedef cnp.float32_t DTYPE_t
''' using the cython styling of code keeping some list elements'''
def convertToPointCloud(cnp.ndarray arr, float cx, float cy, float fx, float fy):
    cdef float depth_scale = 1000.0
    cdef int height = arr.shape[0]
    cdef int width = arr.shape[1]
    pcl_list:list = []
    cdef float x,y,z
    cdef int u, v
    for d in range(height*width):
        u = d%width
        v = d//width
        z = -arr[v,u]/depth_scale
        x = ((<float> u - cx) * z)/fx
        y = ((<float> v - cy) * z)/fy
        pcl_list.append([x, y, z])
    return np.array(pcl_list, dtype=np.float32)


'''copied python code '''
def standardConvertToPointCloud(arr:np.ndarray, cx:float,cy:float,fx:float,fy:float)->np.ndarray:
    depth_scale:float = 1000.0
    height, width = arr.shape
    def indexConversion(inp, width=width,height=height):
        return (inp%width,inp//width)
    pcl_list = []
    for d in range(height*width):
        u,v = indexConversion(d)
        z = -arr[v,u]/depth_scale
        x = ((u - cx) * z)/fx
        y = ((v - cy) * z)/fy
        pcl_list.append([x, y, z])
    return np.array(pcl_list,dtype=np.float32)
```


To compare the methods we are have a base image of size240,320 we run the method on it to process the image and create the pointcloud.  We use timeit package in python with repeat 100 as a This is run 100 times and each timing is recorded and averaged. then we run it through scaling the image to check how the img scales to collect the size of the image on runtime.  

| method | 1x img scale | 2x scale | 4x scale |
| ------ | ------------ | -------- | -------- |
| pure python |  0.073006     |  .29977  | 1.21954  |
| python compiled|0.051608 | 0.21764 | 0.89365  |
| cython typing| 0.030542 | 0.137411 | 0.63505  |

What we see in the chart is that thir are are stark difference in runtime between the Cython versions and the pure python version.  we see an approximately 1/3 decrease in the runtime of the python compiled implementation and 1/2 decrease in the cython implementation. Then we notice that their is a linear growth in the runtime of pixel count and runtime for all method. This makes sense as it has all methods have O(n) runtime.


## static allocation and memory views
The Test is ran in a similar fashion except std_deviations are given to provide insight in to differences of run time.  A cchange to the code was made we use numpy to allocate an array of zeros and we change the values and return the array instead of using a list and to which is then put into an numpy array and returned.

```python 
@cython.boundscheck(False)
@cython.wraparound(False)
def convertToPointCloud(cnp.ndarray arr, float cx, float cy, float fx, float fy):
    depth_scale:float = 1000.0
    height = arr.shape[0]
    width = arr.shape[1]
    def indexConversion(inp, width=width,height=height):
            return (inp%width,inp//width)
    pcl_list = np.zeros((height*width,3), dtype=np.float32) # code change pre allocating size of returned array
    cdef float x, y, z
    cdef int u, v
    for d in range(height*width):
        u = d%width
        v = d//width
        z = -arr[v,u]/depth_scale
        pcl_list[d//3, 0] = = ((<float> u - cx) * z)/fx
        pcl_list[d//3, 1] = ((<float> v - cy) * z)/fy
        pcl_list[d//3, 2] = z
    return np.array(pcl_list,dtype=np.float32)
```


| method | 1x img scale | dev | 2x scale | dev | 4x scale | dev |
| ------ | ------------ | --- | -------- | --- | -------- | --- |
| cython w/list | 0.030542 | .0074 | 0.137411 | 0.0093 | 0.63505  | 0.015 |
| cython static alloc| 0.030632 | 0.0004 | 0.130141 | 0.0044 | 0.56791  |0.0138 |


What we can see is the static allocation is slower at 1x image scale but .1 ms this is most likely due to the large allocation of memory instead of going with the append. in the larger scales we see a speed up with a 5% reduction in runtime and a at the 4x image scale we see an ever decrease in runtime of about 11%. this is big at makin our code faster. This is great but we still are not as we have cut runtime down drastically by 50%.  There is still more to do as we still not have unlocked true speed speed ups.  Then we run into the issues we are using numpy arrays not memviews which is essentially just the raw memory. With This we can open up some we reduce the amount of code created as memory views are not a python object but a memory which can reduce processing speeds.

```python
def convertToPointCloudMem(cnp.ndarray arr, float cx, float cy, float fx, float fy) -> np.ndarray:
    cdef float[:,:] d_arr = arr
    cdef float depth_scale = 1000.0
    cdef int height = arr.shape[0]
    cdef int width = arr.shape[1]
    cdef float[:,:] pcl_list = np.zeros((height*width,3), dtype=np.float32)
    cdef float x, y,z
    cdef int u, v, ind, x_index, y_index, z_index
    x_index = 0
    y_index = 1
    z_index = 2
    for d in range(height*width):
        ind = d//3
        u = d % width
        v = d // width
        z = -d_arr[v,u]/depth_scale
        pcl_list[ind, x_index] = ((<float> u - cx) * z)/fx
        pcl_list[ind, y_index] = ((<float> v - cy) * z)/fy
        pcl_list[ind, z_index] = -d_arr[v,u]/depth_scale
    return np.array(pcl_list,dtype=np.float32)

```

| method | 1x img scale | dev | 2x scale | dev | 4x scale | dev |
| ------ | ------------ | --- | -------- | --- | -------- | --- |
| cython static alloc numpy| 0.030632 | 0.0004 | 0.130141 | 0.0044 | 0.56791  |0.0138 |
| memory view implke| 0.00610 | 0.0012  | 0.0228923 | 0.00306 | 0.0885817 | 0.00028 |

This is big we see a 5x to 7x runtime reduction in speed which is great, but there is more we can do duch as doing something yoiu normally can't in python multi-threading.


### parralelizing the code

In python 3.12 we lack true multithreading through the use of the GIL. But We can disable it in cython this and this can be doen with the cython.parralel which contains a lot of utils to multithread the coding but the one were gonna use is prange. with the nogil parameter set to true(dows not work unless you do in python 3.12). This allows to use the multi threaded paradigm as a way to drop runtime drastically. Outside of the this you can also use parallel class as a way to run code with nogil set To True using the parrallel class, Since this not the best method for our case we will not use it but it is another method to solve the same issue.

```python
from cython.parallel import prange 


@cython.cdivision(True)
def convertToPointCloudMem(cnp.ndarray arr, float cx, float cy, float fx, float fy) -> np.ndarray:
    cdef float[:,:] d_arr = arr
    cdef float depth_scale = 1000.0
    cdef int height = <int> arr.shape[0]
    cdef int width = <int> arr.shape[1]
    cdef float[:,:] pcl_list = np.zeros((height*width,3), dtype=np.float32)
    cdef float z
    cdef Py_ssize_t d # added as this is required for using division
    cdef int u, v, ind
    cdef int x_index, y_index, z_index
    x_index = 0
    y_index = 1
    z_index = 2
    for d in prange(height*width, nogil=True):# changing out range with prange from the 
        ind = <int> d/3
        u = <int> d % width
        v = <int> d / width
        z = -d_arr[v,u]/depth_scale
        pcl_list[ind, x_index] = ((<float> u - cx) * z)/fx
        pcl_list[ind, y_index] = ((<float> v - cy) * z)/fy
        pcl_list[ind, z_index] = -d_arr[v,u]/depth_scale
    return np.array(pcl_list,dtype=np.float32)
```

| method | 1x img scale | dev | 2x scale | dev | 4x scale | dev |
| ------ | ------------ | --- | -------- | --- | -------- | --- |
| memory view | 0.00610 | 0.0012  | 0.0228923 | 0.00306 | 0.0885817 | 0.00028 |
| memory view parallel| 0.001177 | 0.000075  | 0.0043309 | 0.00012 | 0.0126539 | 0.00028 |

we've gotten our code to a point where it is 73x to 100x faster through which is enought to sort our use case but their are speed ups still available such as simd. I am happy wiht this performance as i have taken my dpeth to PCL generation code from being the major computational blocker in converting a depth image to a PointCloud. This leaves more room to use a more computationaly intense model model that can reduce blocks and latency in robot operation.emote --merge       run: "[[ -f package-lock.json || -f npm-shrinkwrap.json ]] && npm ci || true"
