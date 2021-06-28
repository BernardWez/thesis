# An exploration on the effectiveness of multilingual geotagging approaches

## Repository Structure

    .
    ├── data                            # contains all dataset files
    |   ├ CoNLL-2002                    
    |   ├ CoNLL-2003                  
    |   ├ DutchPolicyDocs
    |   |   ├── DutchPolicyDocs.json         # contains final annotated dataset
    |   |   ├── scraping                     # contains notebooks to scrape policy documents
    |   |   ├── visie-documents              # contains intermediate datasets
    |   |   ├ annotated                             # extracted paragraphs with annotations
    |   |   ├ manual-extraction                     # extracted paragraphs from PDF's
    |   |   └ pdf                                   # scraped PDF policy documents
    |   ├ GeoWebNews                     
    |   ├ LGL             
    |   └ TR-News                           
    ├── experiments                     # contains all model notebooks & code files
    |   ├ LaBSE                             
    |   ├ XLM-RoBERTa                 
    |   ├ flair
    |   ├ mBERT             
    |   ├ spaCy
    |   └ utils                             # contains all utility functions
    ├── results                         # contains result tables
    └── README.md
