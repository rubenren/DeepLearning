def create_submission(predictions, fname="submission.csv"):
    """
            `predictions` - numpy array of dimension (n_samples,)
    """
    with open(fname, 'w') as f:
        f.write("ImageId,Class")
        for idx,yhat in enumerate(predictions):
            f.write("\n{},{}".format(idx,yhat))