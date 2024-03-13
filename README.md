# README

## Purpose

Motion compensation of focal observations in contrast-enhanced ultrasound (CEUS) videos


## Publication

Tiyarattanachai T, Turco S, Eisenbrey JR, Wessner CE, Medellin-Kowalewski A, Wilson S, Lyshchik A, Kamaya A, Kaffas AE. A Comprehensive Motion Compensation Method for In-Plane and Out-of-Plane Motion in Dynamic Contrast-Enhanced Ultrasound of Focal Liver Lesions. Ultrasound Med Biol. 2022 Nov;48(11):2217-2228. doi: 10.1016/j.ultrasmedbio.2022.06.007. Epub 2022 Aug 13. PMID: 35970658; PMCID: PMC9529818.


## How to get the data

"sample-data.zip" contains synthesized sample data, which can be used to test the codes. Due to certain restrictions in data sharing policy, the data from real cases are not available publicly.


## How to run

### prepare test sample data

extract the "sample-data.zip" file. \
Within the "sample-data" folder, you will find:
1. "P_001" folder, which represents a case. \
   1.1 "pickle_bmode_CE_gray" folder containing a ".pkl" file. This is a numpy array of CEUS video data having a shape of (frames, height, width), stored as Python Pickle format. \
   1.2 "nifti_segmentation" folder containing a ".nii.gz" file. This is segmentation of the lesion in a reference frame. The segmentation was created using ITK-SNAP software (http://www.itksnap.org/pmwiki/pmwiki.php). Please note that the nibabel python package loads the segmentation file as an array of shape (width, height, frame), so it needs to be transposed before matching with the .pkl file. The motion compensation codes already handle this transpose operation.
2. "dataset_synthesize.xlsx" spreadsheet. The codes access data of each case through this spreadsheet. Apart from paths to the .pkl and .nii.gz files, the spreadsheet contains the following coordinate information using the upper-left corner as the origin (0,0): \
y0_bmode: topmost y-coordinate of the B-mode window \
y0_CE: topmost y-coordinate of the contrast-enhanced window \
x0_bmode: leftmost x-coordinate of the B-mode window \
x0_CE: leftmost x-coordinate of the contrast-enhanced window \
h_bmode: height of the B-mode window \
h_CE: height of the contrast-enhanced window \
w_bmode: width of the B-mode window \
w_CE: width of the contrast-enhanced window \
CE_window_left(l)_or_right(r): indicating whether the contrast-enhanced window is on the left (l) or right (r) side \
If the raw data are DICOM files, the coordinate information can be extracted from the DICOM header. The coordinates are necessary for translating motion-compensated ROIs from B-mode to contrast-enhanced window.

### install packages

conda env create -f environment.yml -n ILSA

### run motion compensation codes

conda activate ILSA \
Then, run "motion-compensation.ipynb" within the "code" folder. \
As described in the paper, the "ROI_margin_suffix_list" variable determines whether to include a margin to the ROI during tracking, and the "frame_rate_suffix_list" variable determines whether to use all frames or half of the frames for tracking.

### outputs

"outputs.zip" contains the expected outputs:
1. "withoutROImargin_fullFrameRate" folder containing motion-compensated ROIs of B-mode (bmode_MC_ROI.pkl) and contrast mode (CE_MC_ROI.pkl), having a shape of (frames, height of ROI, width of ROI). Please note that the folder name will change according to the "ROI_margin_suffix_list" and "frame_rate_suffix_list" settings.
2. "mp4-outputs" for convenient visualization of motion compensation results
