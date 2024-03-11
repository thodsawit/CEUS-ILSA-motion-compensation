# README

## Authors

Thodsawit Tiyarattanachai


## Publication

Tiyarattanachai T, Turco S, Eisenbrey JR, Wessner CE, Medellin-Kowalewski A, Wilson S, Lyshchik A, Kamaya A, Kaffas AE. A Comprehensive Motion Compensation Method for In-Plane and Out-of-Plane Motion in Dynamic Contrast-Enhanced Ultrasound of Focal Liver Lesions. Ultrasound Med Biol. 2022 Nov;48(11):2217-2228. doi: 10.1016/j.ultrasmedbio.2022.06.007. Epub 2022 Aug 13. PMID: 35970658; PMCID: PMC9529818.


## File structure

Combining resources across OSF and GitHub should yield the following structure.

```
├── .gitignore          <- Lists files to be ignored in syncing between local and remote.
├── LICENSE             <- Describes license to the contents of this repo.
├── README.md           <- Describes the project and orchestration (how to run)
│
├── data
│   ├── raw             <- The original, immutable data dump.
│   │   └── <experiment>
│   │       └── <conditions/replicate>
│   │           └── <date as YYYY-MM-DD>
│   ├── external        <- Data from third party sources (e.g., US Census).
│   │   └── <provider>
│   │       └── <date as YYYY-MM-DD>
│   ├── intermediate       <- Intermediate data that has been standardized, formatted, deduped, etc.
│   │   └── <experiment>
│   │       └── <conditions/replicate>
│   │           └── <date as YYYY-MM-DD>
│   ├── extracted       <- Tabular data extracted from conformed image data.
│   │   └── <experiment>
│   │       └── <conditions/replicate>
│   │           └── <date as YYYY-MM-DD>
│   └── tidy            <- The final, canonical datasets for analysis. Includes engineered features.
│       └── <experiment>
│
├── code
│   ├── data-processing <- Code to process data from raw all the way to tidy.
│   │   └── <experiment>
│   ├── draft-analyses  <- Code that operates on tidy data for draft data analytics and visualizations.
│   └── final-analyses  <- Code that operates on tidy data to produce text, figures and tables as they appear in pubilcations.
│
├── output
│   ├── draft           <- Tables and figures from the draft analytics
│   │   └── <experiment>
│   └── final           <- Tables and figures from the final analytics
│
├── docs                <- Data dictionaries, manuals, and all other explanatory materials.
│
├── publication                      
│   └── journal                      <- Journal that this was submitted to
│       └── submission-1_YYYY-MM-DD  <- All materials of submission 1
│           ├── docs                 <- All documents for submission
│           ├── figures              <- All figures for submission
│           └── tables               <- All tables for submission
│
├── .github
│   ├── linters         <- Configuration files for all linters being used
│   └── workflows       <- GitHub Actions workflows
│
├── Dockerfile          <- Use docker build to build Docker container based on this file
├── deps.R              <- Import packages not used elsewhere to help renv
├── renv.lock           <- Lockfile with all dependencies managed by renv
└── renv                <- Package dependency management directory with renv
```


## Code structure

All code follows the following structure.

```
├── Title
│   ├── Inputs          <- Define the input sources.
│   └── Outputs         <- Define the outputs.
│
├── Setup
│   ├── Import          <- Import modules.
│   ├── Parameters      <- Input parameters (e.g., data definitions)
│   ├── Configs         <- Input any configurations (e.g., source data, % sampled).
│   └── Functions       <- Define all functions.
│
├── Read
│   ├── Import          <- Import data.
│   └── Conform         <- Conform data to a format appropriate for analysis.
│
├── Compute
│   └── Compute - <Analysis type>   <- Compute descriptive statistics, visualize, analyze.
│       └── <Analysis subtype>      <- Analysis subtype (if applicable; e.g., histograms).
│
├── Write
│   ├── Conform         <- Conform data to a format appropriate for storage.
│   └── Export          <- Write/push/sink data to a storage service.
│
├── Reproducibility
│   ├── Linting and styling
│   ├── Dependency management
│   └── Containerization
│
├── Documentation
    ├── Session info
    └── References
```


## How to get the data




## How to run

### install packages

conda create --name ILSA --file requirements.txt -c conda-forge \
conda activate ILSA \
pip install opencv-python==4.5.5.64

### run motion compensation codes

conda activate ILSA \
jupyter lab \
Then, run "motion-compensation.ipynb" within the "code" folder.


## How to get help

If you encounter a bug, please file an issue with a minimal reproducible example [here](https://github.com/serghiou/repo-template/issues) and please Label it as a "bug" (option on the right of your window). For help on how to use the package, please file an issue with a clearly actionable question [here](https://github.com/serghiou/repo-template/issues) and label it as "help wanted." For all other questions and discussion, please email the first author.


## How to contribute

1. Create an issue as described above.
2. Create a branch from the issue as described [here](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-a-branch-for-an-issue).
3. Branch names should use the format and voice of this example: "152-bugfix-fix-broken-links".
4. Issue a pull request to initiate a review.
5. Merge using "Rebase and merge" after you've squashed all non-critical commits.


## Be a good citizen

If you like or are reusing elements of this repo, please give a star!
