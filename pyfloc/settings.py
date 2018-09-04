colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'indigo', 'silver', 'tomato', 'gold', 'springgreen', 'tan', 'cadetblue', 'aqua', 'khaki', 'indianred', 'brown', 'lime', 'ivory', 'lightsalmon', 'teal']
numerical_precision = 1e-10
def gaussian(x, mu, sigma):
    y = (1.0 / (sigma*np.sqrt(2.0*np.pi))) * np.exp(-0.5*np.power((x - mu)/sigma, 2.0))
    return y
