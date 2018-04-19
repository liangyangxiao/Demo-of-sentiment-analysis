# Demo-of-sentiment-analysis
Demo-Sentiment Analysis of Movie Review based on SVM

# Detailed Instruction on Demo

 - File List:<br/>
   1, tkinter_test.py<br/>
   2, sentiment_analysis.py<br/>
   3, imdb_labelled.txt<br/>
   4, yelp_labelled.txt<br/>
   5, amazon_cells_labelled.txt<br/>
<br/>
 - Parameter<br/>
    Python version: 3.6<br/>
    Python IDE: Spyder<br/>
<br/>
 - Python Main Extra Package needed list:<br/>
    1, nltk package<br/>
    2, sklearn package<br/>
    3, pandas package<br/>
    4, tkinter package<br/>
    (More details are listed on the beginning of program files)
<br/><br/>
 - How to run the demo<br/>
   1, Please make sure that tkinter_test.py, sentiment_analysis.py, imdb_labelled.txt these 3 files in the same folder.<br/>
   2, Open tkinter_test.py in IDE and run it.<br/>
   3, Now you can see the 'Sentiment Analysis Demo' window showed on the screen.<br/>
   4, Input the review that you want to test into the empty field. e.g. 'This is the best movie I have ever seen.'<br/>
   5, Click 'SUBMIT' button and you can see the sentiment analysis result show in the black block.'pos' result means 'positive' and 'neg' result means 'negtive'.<br/>
<br/>(This demo will not only be limited to movie review analysis, you can change 'imdb_labelled.txt' file to 'yelp_labelled.txt' or 'amazon_cells_labelled.txt' from the code line 30 in sentiment_analysis.py)
<br/>
<br/><br/>
 - Reference<br/>
    for dataset we used:<br/>
    https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences<br/>
    for 'high_information_feature' part in sentiment_analysis.py:<br/>
    https://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/<br/>


