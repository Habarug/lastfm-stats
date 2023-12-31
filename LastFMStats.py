import calendar
import os
import urllib
from datetime import timedelta

import coverpy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


class LastFMStats:
    __font_size_axis_labels = 20
    __font_size_title = 26
    __font_size_ticks = 14
    __font_size_legend = 18
    __fig_size = (15, 7)

    def __init__(self, lastFMExportFile, year=None):
        self.__year = year
        self.__import_data(lastFMExportFile)
        self.__coverpy = coverpy.CoverPy()  # instantiate coverpy

    def __import_data(self, filename):
        self.scrData = self.__read_scrData(filename)

        if isinstance(self.__year, int):
            self.scrData = self.scrData[self.scrData["time"].dt.year == self.__year]

        self.topArtists = self.scrData["artist"].value_counts()[:].index.tolist()
        self.topAlbums = self.scrData["album"].value_counts()[:].index.tolist()
        self.topTracks = self.scrData["track"].value_counts()[:].index.tolist()

        self.__dates = pd.date_range(
            self.scrData["date"].iloc[-1],
            self.scrData["date"].iloc[1] - timedelta(days=1),
            freq="d",
        )

        self.__nScrobbles = self.scrData["artist"].size

        self.__week = self.__weekday_count(
            self.scrData["date"].iloc[-1], self.scrData["date"].iloc[1]
        )

    def __read_scrData(self, filename):
        scrData = pd.read_csv(
            filename, header=None, names=["artist", "album", "track", "time"]
        )
        scrData["time"] = pd.to_datetime(scrData["time"])
        scrData["date"] = scrData["time"].dt.date
        scrData["date"] = pd.to_datetime(scrData["date"])
        scrData["wd"] = scrData["time"].dt.weekday
        scrData["artist"].size

        return scrData

    def __weekday_count(self, start, end):
        week = {}
        for i in range((end - start).days):
            day = calendar.day_name[(start + timedelta(days=i + 1)).weekday()]
            week[day] = week[day] + 1 if day in week else 1
        return week

    def plot_artist_album_distribution(
        self, nArtists=15, artLimitCoefficient=0.05, saveFig=False
    ):
        coverdir = "covers"

        if not os.path.exists(coverdir):
            os.mkdir(coverdir)

        sizeX = 20
        sizeY = 8
        plt.rcParams["figure.figsize"] = (sizeX, sizeY)
        width = 0.9
        cover = 0.85

        artLimit = (
            artLimitCoefficient
            * self.scrData[self.scrData["artist"] == self.topArtists[0]].shape[0]
        )
        scale = (
            self.scrData[self.scrData["artist"] == self.topArtists[0]].shape[0]
            / nArtists
            * sizeX
            / sizeY
        )

        fig3, ax3 = plt.subplots()

        for artist in self.topArtists[0:nArtists]:
            filterArtist = self.scrData[(self.scrData["artist"] == artist)]
            albums = filterArtist["album"].unique()

            albumsCount = []

            for album in albums:
                albumScrobbles = filterArtist[filterArtist["album"] == album]
                albumsCount.append(albumScrobbles.shape[0])

            albums = [y for x, y in sorted(zip(albumsCount, albums), reverse=True)]
            albumsCount = [x for x, y in sorted(zip(albumsCount, albums), reverse=True)]

            bottom = 0
            plt.gca().set_prop_cycle(None)

            for idx, count in enumerate(albumsCount):
                bp = plt.bar(artist, count, width, bottom=bottom)

                patch = bp.patches
                (x, y) = patch[0].get_xy()

                if count >= artLimit:
                    size = min(count * 0.95, cover * scale)
                    extent = [
                        x + width / 2 - size / (2 * scale),
                        x + width / 2 + size / (2 * scale),
                        bottom + count / 2 - size / 2,
                        bottom + count / 2 + size / 2,
                    ]
                    album = albums[idx][0:30] if len(albums[idx]) > 30 else albums[idx]
                    artistAlbum = artist + " " + album
                    savePath = os.path.join(coverdir, artist + "_" + album + ".jpg")
                    try:  # try local folder first
                        a = mpimg.imread(savePath)
                        ax3.set_autoscale_on(False)
                        plt.imshow(a, extent=extent, aspect="auto", zorder=3)
                    except:
                        try:
                            art = self.__coverpy.get_cover(artistAlbum)
                            f = urllib.request.urlretrieve(art.artwork(100), savePath)
                            a = mpimg.imread(savePath)
                            ax3.set_autoscale_on(False)
                            plt.imshow(a, extent=extent, aspect="auto", zorder=3)
                        except:
                            try:
                                art = self.__coverpy.get_cover(
                                    artist + " " + album.split(" ")[0]
                                )
                                f = urllib.request.urlretrieve(
                                    art.artwork(100), savePath
                                )
                                a = mpimg.imread(savePath)
                                ax3.set_autoscale_on(False)
                                plt.imshow(a, extent=extent, aspect="auto", zorder=3)
                            except:
                                try:
                                    art = self.__coverpy.get_cover(album)
                                    f = urllib.request.urlretrieve(
                                        art.artwork(100), savePath
                                    )
                                    a = mpimg.imread(savePath)
                                    ax3.set_autoscale_on(False)
                                    plt.imshow(
                                        a, extent=extent, aspect="auto", zorder=3
                                    )
                                except:
                                    print("Art not found for " + artistAlbum)

                bottom += count

        plt.xticks(rotation=45, fontsize=self.__font_size_ticks)
        plt.yticks(fontsize=self.__font_size_ticks)
        plt.ylabel("Scrobble count", fontsize=self.__font_size_axis_labels)
        plt.xlim(-0.5, nArtists - 1.5)
        plt.ylim(0, self.scrData[self.scrData["artist"] == self.topArtists[0]].shape[0])
        fig3.patch.set_facecolor("xkcd:white")
        plt.tight_layout()

        if saveFig:
            if not os.path.exists("output"):
                os.mkdir("output")
            plt.savefig(os.path.join("output", "topArtists_new.jpg"), dpi=600)

    def plot_top_artists_timeline(self, nArtists=15, saveFig=False):
        fig, ax = plt.subplots(1)
        plt.rcParams["figure.figsize"] = self.__fig_size

        for artist in self.topArtists[0:nArtists]:
            dailyScrobbles = []
            cumScrobbles = []
            filterArtist = self.scrData[(self.scrData["artist"] == artist)]

            for date in self.__dates:
                filterDate = filterArtist[(filterArtist["date"] == date)]
                dailyScrobbles.append(filterDate.shape[0])
                cumScrobbles.append(sum(dailyScrobbles))

            plt.plot(self.__dates, cumScrobbles)

        plt.legend(
            self.topArtists[0:nArtists], loc="center left", bbox_to_anchor=(1, 0.5)
        )
        plt.ylabel("Total scrobbles", fontsize=self.__font_size_axis_labels)
        fig.patch.set_facecolor("xkcd:white")
        plt.tight_layout()

        if saveFig:
            if not os.path.exists("output"):
                os.mkdir("output")
            plt.savefig(os.path.join("output", "topArtists_timeline_new.jpg"), dpi=600)

    def plot_top_albums_timeline(self, nAlbums=15, saveFig=False):
        fig, ax = plt.subplots(1)
        plt.rcParams["figure.figsize"] = self.__fig_size

        for album in self.topAlbums[0:nAlbums]:
            dailyScrobbles = []
            cumScrobbles = []
            filterAlbum = self.scrData[(self.scrData["album"] == album)]

            for date in self.__dates:
                filterDate = filterAlbum[(filterAlbum["date"] == date)]
                dailyScrobbles.append(filterDate.shape[0])
                cumScrobbles.append(sum(dailyScrobbles))

            plt.plot(self.__dates, cumScrobbles)

        plt.legend(
            self.topAlbums[0:nAlbums], loc="center left", bbox_to_anchor=(1, 0.5)
        )
        plt.ylabel("Total scrobbles", fontsize=self.__font_size_axis_labels)
        fig.patch.set_facecolor("xkcd:white")
        plt.tight_layout()

        if saveFig:
            if not os.path.exists("output"):
                os.mkdir("output")
            plt.savefig(os.path.join("output", "topAlbums_timeline_new.jpg"), dpi=600)

    def plot_WD_distribution(self, nArtists=10, saveFig=False):
        wdCount = []
        wdNums = np.arange(0, 7, 1)

        wds = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        fig, ax = plt.subplots()

        for wd in wdNums:
            filterWd = self.scrData[(self.scrData["wd"] == wd)]
            wdCount.append(filterWd["wd"].size / self.__week[wds[wd]])

            topArtistsWD = filterWd["artist"].value_counts()[:].index.tolist()

            artistStr = ""

            for idx, artist in enumerate(topArtistsWD[0:nArtists]):
                if len(artist) > 16:
                    artist = artist[0:14] + "..."

                artistStr = artistStr + str(idx + 1) + ":" + artist + "\n"
            artistStr = artistStr[0:-1]

            t = ax.text(
                (wd - 0.4) * 4,
                self.__nScrobbles / (sum(self.__week.values()) * 4),
                artistStr,
                fontsize=self.__font_size_ticks,
                rotation=0,
            )
            t.set_bbox(dict(facecolor="white", alpha=0.7, edgecolor="black"))

        ax.bar(wdNums * 4, wdCount)
        ax.set_xticks(wdNums * 4, wds)
        ax.tick_params(axis="both", labelsize=self.__font_size_ticks)
        ax.set_ylabel("Average daily scrobbles", fontsize=self.__font_size_axis_labels)
        fig.patch.set_facecolor("xkcd:white")
        plt.tight_layout()

        if saveFig:
            if not os.path.exists("output"):
                os.mkdir("output")
            plt.savefig(os.path.join("output", "WD_distribution_new.jpg"), dpi=600)

    def plot_stacked_timeline(self, nArtists=15):
        fig, ax = plt.subplots(1)
        plt.rcParams["figure.figsize"] = self.__fig_size

        # All other artists first
        filterArtist = self.scrData[
            self.scrData["artist"].isin(self.topArtists[nArtists:-1])
        ]

        dailyScrobbles = np.zeros([nArtists + 1, len(self.__dates)], dtype=int)
        cumScrobbles = np.zeros([nArtists + 1, len(self.__dates)], dtype=int)
        leg = ["other"]

        for idx, date in enumerate(self.__dates):
            filterDate = filterArtist[filterArtist["date"] == date]
            dailyScrobbles[0, idx] = filterDate.shape[0]
            cumScrobbles[0, idx] = sum(dailyScrobbles[0, 0:])

        for ida, artist in enumerate(reversed(self.topArtists[0:nArtists])):
            filterArtist = self.scrData[self.scrData["artist"] == artist]
            leg.append(artist)

            for idx, date in enumerate(self.__dates):
                filterDate = filterArtist[filterArtist["date"] == date]
                dailyScrobbles[ida + 1, idx] = filterDate.shape[0]
                cumScrobbles[ida + 1, idx] = sum(dailyScrobbles[ida + 1, 0:idx])

            cumScrobbles[ida + 1, :] += cumScrobbles[ida, :]

        for idx, row in enumerate(np.flip(cumScrobbles, 0)):
            ax.fill_between(self.__dates, row, zorder=idx)

        ax.legend(
            np.flip(leg),
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=self.__font_size_legend,
        )
        ax.set_ylabel("Total scrobbles", fontsize=self.__font_size_axis_labels)
        ax.xaxis.set_tick_params(labelsize=self.__font_size_ticks)
        ax.yaxis.set_tick_params(labelsize=self.__font_size_ticks)
        fig.patch.set_facecolor("xkcd:white")

    def __moving_average(self, a, n=30):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    def plot_moving_average_timeline(self, window=30):
        fig, ax = plt.subplots(1)
        plt.rcParams["figure.figsize"] = self.__fig_size
        legText = ["Daily scrobbles", "Moving average, window = %i days" % window]

        dailyScrobbles = []
        for idd, date in enumerate(self.__dates):
            filterDate = self.scrData[self.scrData["date"] == date]
            dailyScrobbles.append(len(filterDate))

        scrobblesMovingAverage = self.__moving_average(dailyScrobbles, window)
        datesMovingAverage = self.__dates[
            math.floor(window / 2) - 1 : -math.ceil(window / 2)
        ]

        ax.plot(self.__dates, dailyScrobbles, color="grey", linestyle="--")
        ax.plot(
            datesMovingAverage,
            scrobblesMovingAverage,
            linewidth=4,
        )

        ax.set_ylabel("Daily Scrobbles", fontsize=self.__font_size_axis_labels)
        ax.xaxis.set_tick_params(labelsize=self.__font_size_ticks)
        ax.yaxis.set_tick_params(labelsize=self.__font_size_ticks)

        ax.legend(
            legText,
            loc="upper center",
            fontsize=self.__font_size_legend,
        )
