=======================================================================================
PLEASE READ THIS INSTRUCTIONS BEFORE EXECUTING ANY FILE
=======================================================================================

This project is composed of the following files:

- processImages.cpp: This is a script that will read all database and query images, and
	calculates their feature fectors, saving them in the "database_feat_vector" and
	"query_feat_vector" folders (be sure that they exist inside the project folder).

- main.cpp: This is a script that will process one or many query images, and show the
	average of the normalized rank obtained for all query images processed, and also
	gives the option to visualize the top 10 results obtained for each query image

- img_databae: Folder of all images that compose the database

- img_query: Folder of all query images that can be used for this project

- database_feat_vector: Folder of all feature vectors obtained for each database image
	(initially empty)

- query_feat_vector: Folder of all feature vectors obtained for each query image
	(initially empty)


REQUIREMENTS:
- OpenCV3+
- CMake 3.5.1+


HOW TO BUILD:

1) Create a new file called "build" in the project directory

2) Open a new terminal and navigate to the "build" folder created in the previous step

3) In the terminal, type: 
	$ cmake ../

4) Finally, type:
	$ make

5) Now, if everything worked, the source files should have been built successfully and the
executables should be located inside the "build" folder


HOW TO EXECUTE:
Once you have the executables, they can be opened through terminal by typing ./processDB or
./proyectoImagenes if you are inside the "build" folder. Additionally, each file has its 
own set of parameters that can be called to get a specific behaviour:

- processDB:
	1) nRects (positional argument): Number of rectangles to use for region division.
		Must be one of 9, 16, 36, or 0 (for non-rectangular division) (0 by default)
	2) dbFolder (positional argument): Folder containing database images (../img_database
		by default)
	3) dbVectfolder (positional argument): Folder used to save feature vectors of database
		images (default is ../database_feat_vector)
	4) queryFolder (positional argument): Folder containing query images (../img_query by
		default)
	5) queryVectFolder (positional argument): Folder used to save feature vectors of query
		images (default is ../query_feat_vector)
	6) --texture: Set this flag to use texture-based descriptor instead of color histogram
		descriptor (this forces nRects to 16)

- proyectoImagenes:
	1) queryVectFolder (positional argument): Folder or single file containing feature
		vector(s) of query image(s) that want(s) to be consulted. (default is 
		../query_feat_vector)
	2) dbVectFolder (positional argument): Folder containing feature vectors of database
		images (default is ../database_feat_vector)
	3) queryImgFolder (positional argument): Folder of query images (default is ../img_query)
	4) dbImgFolder (positional argument): Folder of database images (default is ../img_database)
	5) --showResults: Set this flag to show top 10 results for each query image consulted.
		Results are displayed with in 10 images, where each one will have the query image
		on the left, and the corresponding database image on the right.

All this information can also be found if the flag --help, -h, or -? is set on any of the executables.


EXAMPLES:
To execute with default parameters:
	$ ./processDB
	$ ./proyectoImagenes

To set your own parameters:
	$ ./processDB 9 #Set region division of 9 equally-sized rectangles with all other parameters  as default
	$ ./processDB 16 ../my_img_database ../my_database_feat_vector ../my_img_query ../my_query_feat_vector
	$ ./processDB --texture #To use texture-based descriptor

	$ ./proyectoImagenes --showResults #To show first 10 results for each query image
	$ ./proyectoImagenes ../query_feat_vector/100500.xml #Execute on only one query image
	$ ./proyectoImagenes ../my_query_feat_vector ../my_database_feat_vector ../my_img_query ../my_img_database

