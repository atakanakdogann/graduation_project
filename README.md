# Planning of Bin Allocation and Vehicle Routing in Solid Waste Management

**Authors:** Atakan Akdoğan, Hüseyin Emre Tığcı  
**Supervisor:** Prof. Dr. Didem Gözüpek  
**Institution:** Gebze Technical University, Department of Computer Engineering

[![DOI](https://zenodo.org/badge/1006521127.svg)](https://doi.org/10.5281/zenodo.15768836)
![Language](https://img.shields.io/badge/Language-Python-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Project Description

This project addresses the **Waste Bin Allocation and Routing Problem (WBARP)**, a complex, integrated logistics challenge faced by municipalities in solid waste management. The primary goal is to minimize the total system cost by simultaneously optimizing two interconnected decisions:
1.  **Bin Allocation:** Determining the optimal number, type, and placement of waste bins.
2.  **Vehicle Routing:** Determining the optimal service frequencies and daily collection routes for a fleet of vehicles.

The core of the problem lies in the trade-off between these two cost centers. A higher service frequency increases routing costs but lowers the required investment in bin capacity, and vice-versa. This project implements and compares different matheuristic models to find the best balance.

### Based On

The methodologies and problem formulations in this project are heavily based on the following academic paper:
> Hemmelmayr, V. C., Doerner, K. F., Hartl, R. F., & Vigo, D. (2014). *Models and Algorithms for the Integrated Planning of Bin Allocation and Vehicle Routing in Solid Waste Management*. **Transportation Science, 48(1)**, 103-120.

## Models Implemented

We developed and compared three distinct models to evaluate different planning strategies:

1.  **BAFRS (Bin Allocation-First, Route-Second):** A sequential matheuristic approach. It first solves a Mixed-Integer Linear Program (MILP) to minimize bin-related costs and determine service frequencies. It then uses a Variable Neighborhood Search (VNS) heuristic to solve the routing problem based on the fixed plan from the first phase. This model is "routing-blind" in its strategic phase.

2.  **RFBAS (Route-First, Bin Allocation-Second):** A more advanced matheuristic. This model's MILP considers both bin costs and *estimated* routing costs simultaneously. This allows it to make more geographically "aware" and globally efficient decisions about service frequencies, which are then passed to the VNS for final routing.

3.  **Integrated Model (IM):** A full, monolithic MILP that formulates the entire WBARP as a single problem. While theoretically optimal, it is only computationally tractable for very small instances and serves as a benchmark for evaluating the quality of the matheuristic solutions.


## Dataset

We used the real-world data shared by Kadikoy Belediyesi, from Istanbul. The datas we used can be found in following links:

https://www.google.com/maps/d/u/0/viewer?mid=181mmqgMadJTtgb4SsRSz07tUHTXU-Agy&ll=40.98232538542135%2C29.057219295162938&z=14

https://acikveri.kadikoy.bel.tr/dataset/kadikoy-belediyesi-ambalaj-atik-toplama-konteynerleri

## File Structure

```
/
|- bafrs_model.py          # Main script to run the BAFRS model
|- rfbas_model.py          # Main script to run the RFBAS model
|- gui.py                  # PySide6 GUI for visualizing results
|- requirements.txt        # Project dependencies
|- README.md               # This file
```

## Installation and Setup

To run this project, you need a working Python environment and the IBM CPLEX Optimization Studio.

1.  **Clone the Repository:**
    ```bash
    git clone "https://github.com/atakanakdogann/graduation_project/"
    cd graduation_project
    ```

2.  **Install IBM CPLEX:**
    The `docplex` library is a Python API for CPLEX. You must have the full **IBM ILOG CPLEX Optimization Studio** installed on your system. It is available for free for academics and students through the IBM Academic Initiative.

3.  **Set up a Python Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install Dependencies:**
    Install all required Python libraries using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

You can run each model individually from your terminal. Ensure you are in the project's root directory.

-   **To Run the BAFRS Model:**
    ```bash
    python bafrs_model.py
    ```
    This will execute the BAFRS workflow, print the results to the console, and save the final solution to `bafrs_results.json` and route maps to a results folder.

-   **To Run the RFBAS Model:**
    ```bash
    python rfbas_model.py
    ```
    This will execute the RFBAS workflow and save its results to `rfbas_results.json`.

-   **To Launch the Visualization GUI:**
    The GUI can be used to load the `.json` result files and visualize the daily routes interactively.
    ```bash
    python gui.py
    ```

## Key Findings

Our computational experiments on a large-scale, real-world dataset ($n=333$) yielded clear results:

-   **Integrated planning is superior:** The RFBAS model, which considers routing costs during strategic planning, produced a total system cost **~60% lower** than the sequential BAFRS model.
-   **BAFRS is inefficient:** The BAFRS model's "routing-blind" approach resulted in geographically scattered daily plans, leading to significantly higher routing costs and longer VNS computation times.
-   **RFBAS is effective and efficient:** The RFBAS model not only found a much cheaper solution but also ran **more than twice as fast** as BAFRS, because its "smarter" strategic plan provided an easier problem for the VNS to solve.

## Technologies Used

-   **Language:** Python 3.x
-   **Optimization:** IBM ILOG CPLEX & DOcplex library
-   **Data Handling:** NumPy
-   **Visualization:** Matplotlib, PySide6 (for GUI), Leaflet.js (for map rendering)

## Authors

-   Atakan Akdoğan
-   Hüseyin Emre Tığcı
