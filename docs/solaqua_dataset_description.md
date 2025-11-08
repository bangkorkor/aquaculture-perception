# Data Feature (Dataset) description
Taken from [https://data.sintef.no/feature/fe-a8f86232-5107-495e-a3dd-a86460eebef6](https://data.sintef.no/feature/fe-a8f86232-5107-495e-a3dd-a86460eebef6)

This dataset contains ROV data, navigation data, mono camera and stereo camera data from multiple sea trials conducted at full-scale fish farms. The data points include:

- IMU
- Gyroscope
- DVL
- USBL
- Multibeam sonar
- Ping 360 sonar
- Mono camera
- Stereo camera
- Depth
- Pressure
- Temperature

Filename (timestamp)		        Action performed
2024-08-20_13-39-34		Calibration of stereo camera	
2024-08-20_13-40-35		Calibration of stereo camera	
2024-08-20_13-42-51		Calibration of stereo camera	

2024-08-20_13-55-34		Manual control - shallow        **using** Few fish at the end. Net far away
2024-08-20_13-57-42		Manual control - shallow        **using** Some fish here and there. Net is medium away.
2024-08-20_14-16-05	    Manual control - deeper         **using** net labeling
2024-08-20_14-22-12		Manual control - deeper	
2024-08-20_14-24-35		Manual control - deeper
2024-08-20_14-31-29     Manual control - shallow        **using** Few fish before the end. Net is far away
	

Filename (timestamp)		                                                   Action performed
2024-08-20_14-34-07_data to 2024-08-20_18-52-15     Net following

The net following is performed with varying net distances, depths and velocities

Abbreviations:
D0: Initial desired distance to net [m]
D1: Final desired distance to net [m]
Z: Depth [m]
V: Net-relative velocity [m/s], horizontal
Q: Heading-angle offset from net [deg]

The following sets are somewhat inconsistent in terms of net distance:
---------------------------------------------------------------
Filename	                       D0	D1	Z	V	Q
2024-08-20_14-34-07        1.5   1.5  2     0.2    0    **using** for net-labeling
2024-08-20_14-36-22        1.5   1.5  2     0.2    0
2024-08-20_14-38-37        2.0   2.0  2     0.2    0    **using** Few fishes. Medium distance to net. Used for labeling net. 
2024-08-20_14-49-47        2.0   2.0  2     0.2    0
2024-08-20_14-54-52        2.0   2.0  2     0.2    0
2024-08-20_14-57-38        2.0   1.1  2     0.2    0
2024-08-20_15-00-24        1.5   1.5  5     0.2    0
2024-08-20_15-05-53        1.0   1.5  5     0.2    0
2024-08-20_15-09-34        1.5   x      5     0.2    0 (x = 1.8, 2.1, 1.1)
2024-08-20_15-12-51        1.5   1.0  5     0.1    0
2024-08-20_15-14-40        1.4   1.9  5     0.1    0
2024-08-20_15-18-27        1.4   1.4  5     0.3    0
2024-08-20_15-20-29        1.4   1.4  5     0.3    0
---------------------------------------------------------------

The following sets are consistent in terms of net distance
---------------------------------------------------------------
Filename	                       D0	D1	Z	V	Q
2024-08-20_16-34-34		1	1.5	2	0.2	0
2024-08-20_16-37-15		1	1.5	2	0.2	0
2024-08-20_16-39-23		1	1.5	2	0.2	0
2024-08-20_16-43-25		1	1.5	2	0.2	0
2024-08-20_16-45-21		1	1.5	2	0.2	0
2024-08-20_16-51-57		1	1.5	2	0.2	0
2024-08-20_16-47-54		1	1.5	2	0.2	0       **using** net labeling
							
2024-08-20_16-54-36		1	1.5	2	0.1	0       
2024-08-20_16-57-46		1	1.5	2	0.1	0       
2024-08-20_17-02-00		1	1.5	2	0.1	0       **using** stable data of the net. Few fishes, close to new
2024-08-20_17-04-52		0.5	1	2	0.1	0
2024-08-20_17-08-14		0.5	1	2	0.1	0
2024-08-20_17-11-14		0.5	1	2	0.1	0
						
2024-08-20_17-14-36		1	1.5	2	0.3	0       **using** net labeling
2024-08-20_17-22-40		1	1.5	2	0.3	0
2024-08-20_17-31-58		1	1.5	2	0.3	0
2024-08-20_17-34-52		1	1.5	2	0.3	0
2024-08-20_17-37-08		1	1.5	2	0.3	0
							
2024-08-20_17-39-32		1	1.5	5	0.2	0
2024-08-20_17-40-54		1	1.5	5	0.2	0
2024-08-20_17-47-49		0.5	1	5	0.2	0
2024-08-20_17-50-22		0.5	1	5	0.2	0
2024-08-20_17-53-06		0.5	1	5	0.2	0
2024-08-20_17-55-40		1	1.5	5	0.2	0
							
2024-08-20_17-57-55		0.5	1	5	0.1	0
2024-08-20_18-01-46		0.5	1	5	0.1	0
2024-08-20_18-05-42		0.5	1	5	0.1	0
2024-08-20_18-07-47		0.5	1	5	0.1	0
2024-08-20_18-09-52		0.5	1	5	0.1	0
2024-08-20_18-12-20		0.5	1	5	0.1	0
							
2024-08-20_18-38-53		0.5	1	5	0.2	0
2024-08-20_18-41-02		0.5	1	5	0.2	0
							
2024-08-20_18-47-40		1	1	2	0.2	10      **using** its too close
2024-08-20_18-53-59		1	1	2	0.2	10
							
2024-08-20_18-50-22		1	1.5	2	0.2	0   **using** net labeling
2024-08-20_18-52-15		1	1.5	2	0.2	0
---------------------------------------------------------------

The following sets were gathered on another date and includes measurements from two different DVLs, namely the Waterlinked A50 and the Nortek Nucleus 1000. A short description of the actions performed while gathering the datasets are given for each dataset.

NFH = Net following horizontal

2024-08-22_14-06-43 NFH, 2 m depth. Dist. 0.5m to 1 meters. 0.2 m/s speed
2024-08-22_14-29-05 NFH, 2 m depth. Dist. 0.6m to 0.8 meters. 0.1 m/s speed

2024-08-22_14-47-39 NFH, 2 m depth. Dist. 0.6m, no change. 0.1 m/s speed
2024-08-22_14-48-39 NFH, 2 m depth. Dist. 0.6m, no change. 0.1 m/s speed
----------------------------------------------------------------------------------------------------------

The following set contains a change in heading offset and in the direction
----------------------------------------------------------------------------------------------------------
2024-08-22_14-50-14 NFH, 2 m depth. Dist. 0.6m, some change in dist. 0.1 m/s speed. Changing heading offset and direction.
----------------------------------------------------------------------------------------------------------

SOLAQUA_msg_files contains most, if not all, relevant ROS messages for the datasets. 

Environmental data:
- Waves: N/A
- Current: 0.04-0.2 m/s
- Wind: 6 m/s
- Air temperature: 14 C
- Weather: Rain

Biomass data (Approximated values):
- Number of fish: ~188 000
- Average weight: ~3000 grams

Fish cage data:
- Cage height (submerged): -
- Cage diameter (at surface): 50 m
- Cage circumference: 157 m
- Net mesh grid size: 27.5mm x 27.5 mm
- Parts of the net has biofouling