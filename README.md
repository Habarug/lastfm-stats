## LastFM stats

Various ways to plot LastFM scrobble data. Ugly, barely functional, fragile code that is likely to break, and mostly things that have been implemented better elsewhere, but I like the top artists with album distribution as stacked bar plots. 

## Instructions

- Download LastFM data with [Benjamin Ben's lastfm-to-csv](https://benjaminbenben.com/lastfm-to-csv/)
- Instantiate the class with your data (also shown [Jupyter example](LastFM-stats-testing.ipynb)):

    ```python 
    import LastFMStats
    lastfm = LastFMStats.LastFMStats(*your filename*)
    ```
    - Optional: Filter your data to a specific year:
    ```python
    import LastFMStats
    lastfm = LastFMStats.LastFMStats(*your filename*, year = 2023)
    ```
- Plot something!: 
    ```python
    lastfm.plot_artist_album_distribution(
        nArtists = 15,
        artLimitCoefficient = 0.05,
        saveFig = False
    )
    ```
    ![plot](output/topArtists_2023-12-23.jpg)

### Note on album covers

This uses the [coverpy library](https://github.com/matteing/coverpy) to fetch album covers, and stores them in /covers. It mostly works, except when it doesn't. Some covers it just can't find at all. There are probably better solutions, because services like tapmusic.net finds album covers way more consistently. For now, this can worked around by adding missing/wrong covers manually to the /covers folder. Next time you run plot_artist_album_distribution() the local files will be used. 