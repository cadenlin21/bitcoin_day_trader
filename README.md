#Bitcoin Day Trader

This project simulates an algorithm to optimize day trading using bitcoin, cash, and gold as the only assets available. The model incorporates several common threads of basic quantitative investing strategies, such as momentum trading, mean reversion, and noise reduction. In particular, using historical bitcoin and gold price data, the algorithm takes the following steps to determine an optimized trading strategy:

  1. Calculate price gradients, volatility, and profitability 
  2. Control for limited data using noise reduction techniques 
  3. Issue sell signals (if the amount gained from selling, accounting for a commission, subtracted by moving average is above a certain margin, then sell an amount proportional to the amount that would be made by the sell) 
  4. Issue buy signals (if price is rising but below the moving average) and buy an amount proportional to the asset's profitability 
  5. Conduct a sensitivity analysis for various parameters comparing the performance of the model under different conditions, then choose the optimal set of parameters for the final simulation. 
  
The results of the simulation using a principal investment of $10,000 over a 5-year period of simulated trading, using real bitcoin and gold data from 2016 to 2021, are shown. The model turned this investment into $90,872 at the end of the simulation, reflecting a 908.72% ROI. The model can be simulated with other principal investments and other time periods by simply substituting different file inputs (must be CSV in date | price format) and initial cash values.
