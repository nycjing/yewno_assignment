# yewno_assignment

1. Use freely available data from the web to predict/explain macroeconomic indicators.

I got Linkedin Profiles data set (linkedin_data.csv) from Thinknum. I just got it through their demo request. 
I'm interested in the number of employees_on_platform and followers. Maybe by looking into the changes of employees or followers, 
we can spot a turning point, or upward/downward momentume.

2. Smart Beta strategy:

There are a wide varieties of smart beta strategies. In general there are two types of indexes:

	Alternatively weighted indexes — typically designed to address perceived concentration risks in 
					capitalization-weighted indexes or reduce volatility within the index;
	Factor indexes — designed to replicate factor risk premia in a transparent, rules-based and investable format.
	
I personally in favor of alternative weighted benchmarks that, for instance, weight individual stocks equally, 
or according to dividends, company fundamentals... Given Cap weighted indexes are the most popular benchmark weighting scheme, alternative weighting scheme is a way to against herd mentality.  


3. Paris trading strategy
  
I believe in Emerging Market the equity market has a strong connection with 
Sovereign debt index and Commodity index. Based on the OLS regression result I can 
build a trading strategy to capture the trading opportunities. (We can use VAR or other models to capture more time series info.)

code: yewno_pairs_trading.py  
