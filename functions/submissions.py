# import reddit api wrapper
import praw
import pandas as pd
import numpy as np

# create a praw Reddit instance with app credentials
reddit = praw.Reddit(
    client_id="XbesrQBvKymjgLdgg_D6lA",
    user_agent="NFLTextAnalysis/0.0.1",
    username="ta_api"
)

def get_submissions(subreddit:str, limit:int) -> pd.DataFrame:
    """ 
    Takes a subreddit name and how many of the last month's most popular posts to return as a dataframe.

    Returns the title, whether the submission is a self-post or external link, the body of the submission (in html),
    the url (to the external link, if not a self-post, otherwise, to the reddit submission itself),
    the number of upvotes, and the ratio of upvotes to downvotes.
    """
    
    # for each submission returned from the subreddit's top month query, gather neccessary meta-data features
    submission_data = [(submission.id,
                            submission.title,
                            submission.created_utc,
                            submission.is_self,
                            submission.url,
                            submission.ups,
                            submission.upvote_ratio,
                            submission.num_comments) for submission in reddit.subreddit(subreddit).top(time_filter='month', limit=limit)]
    
    # save and return dataframe of submission data
    submission_df = pd.DataFrame(submission_data, columns=['submission_id', 'title', 'created_utc', 'is_self', 
                                                           'url', 'ups', 'upvote_ratio', 'num_comments'])

    return submission_df


def get_subscribers(subreddit: str):
    # because the number of subscribers changes constantly, return the most recent count
    # for a given subreddit
    
    subs = reddit.subreddit(subreddit).subscribers

    return subs
            