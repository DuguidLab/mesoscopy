import click
from matplotlib import pyplot as plt


def plot_line(array, outpath, message=None):
    plt.clf()
    plt.plot(array)
    plt.savefig(outpath)
    if message:
        click.echo(message)


def plot_lines(arrays, outpath, message=None):
    plt.clf()
    for array in arrays:
        plt.plot(array)
    plt.savefig(outpath)
    if message:
        click.echo(message)


def plot_hist(array, outpath, message=None):
    plt.clf()
    plt.hist(array)
    plt.savefig(outpath)
    if message:
        click.echo(message)


def plot_frame(array, outpath, message=None):
    plt.clf()
    plt.imsave(outpath, array)
    if message:
        click.echo(message)


def plot_scatter(x, y, outpath, labels=[], message=None):
    plt.clf()
    plt.scatter(x, y)
    if labels:
        plt.legend(labels)
    plt.savefig(outpath)
    if message:
        click.echo(message)


def plot_scatters(xs, ys, outpath, labels=[], message=None):
    plt.clf()
    for x, y in zip(xs, ys):
        plt.scatter(x, y)
    if labels:
        plt.legend(labels)
    plt.savefig(outpath)
    if message:
        click.echo(message)
