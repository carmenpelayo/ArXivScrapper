# 📈 arXiv Preprints: A Real-Time Proxy of Technological Research Activity

This repository contains Python code to **measure the evolution of research activity worldwide** by counting monthly publications in different arXiv knowledge categories. 

[ArXiv](https://arxiv.org/) is an open-access research paper repository covering a wide range of disciplines, including physics, mathematics, computer science, biology, statistics, quantitative finance, and more. As one of the most important platforms for the early dissemination of research results, arXiv is considered a reliable indicator of global scientific progress.

The objective of this project is to provide Python code that enables the user to easily extract time series of monthly volume of publications in different [arXiv categories](https://arxiv.org/category_taxonomy). These series allow to identify trends, growth patterns, and emerging research areas, providing a valuable database for scientific and strategic analysis.

Additionally, a web app was created to leverage the extracted data: **[link](https://arxivtracker.streamlit.app/)**. It can be used to:
1.  Forecast the future evolution of a certain category.
2.  Numerically and visually analyze different arXiv categories.
3.  Decompose a certain category to analyze its time components.
4.  Build your own index by aggregating categories, smoothing and standardizing.

## Methodology

The project's methodology focuses on:

1. **URL Generation**: Automatically constructing monthly URLs (e.g. `https://arxiv.org/list/cs.AI/2000-04` for AI papers in April 2000) for selected arXiv categories, from a specified start month to an end month.
   
2. **Web Scraping**: Using Python libraries such as `requests` and `BeautifulSoup`, the monthly publication counts available on the corresponding page for each category are extracted.
   
3. **Data Storage**: The collected data is organized into time series with clear identifiers by month and category.

4. **Analysis and Visualization** (optional): The extracted data can be used to generate graphs and models analyzing the behavior of scientific production over time.

## Potential Applications

This project has a wide range of applications:

1. **Econometric Models**: Integrating arXiv time series into econometric models to study the impact of research on the global economy.

2. **Strategic Vision**: Providing academic institutions and research organizations with a tool to identify emerging areas and set priorities.

3. **Trend Monitoring**: Monitoring specific disciplines to evaluate the progress of science and technology quantitatively.

4. **Impact of Scientific Policies**: Analyzing the effect of governmental or private initiatives on the volume of scientific publications.

5. **Visualization of the Scientific Ecosystem**: Creating interactive dashboards for exploring data by category and time period.

## Final Considerations

1. **Scraping Limitations**: Scraping is limited by the availability of data on arXiv and the platform's usage policies. It is recommended not to make excessive requests in a short period.

2. **Extensibility**: This project can be expanded to include more categories, longer time intervals, or integration with other data sources.

3. **Collaboration**: Other developers and researchers are invited to contribute improvements to the code or additional analyses.

---

This project represents an effort to democratize access to scientific data and promote trend analysis in global research. If you have questions or suggestions, feel free to open an issue or submit a pull request.
