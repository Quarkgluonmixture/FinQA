# Error Cases (Auto Extracted)

## 1. parse fail
- question: did altria outperform the s&p 500?
- gold: None
- pred: None
- raw_output: | 2016 | 2015 | 2014 | 2013
altria group inc . | $ 286.61 | $ 225.81 | $ 215.81 | $ 215.81
peer group | 192.56 | 160.81 | 158.81 | 158.81
s&p 500 | 198.09 | 188.95 | 186.75 | 186.75


.

Output:
[FINAL_ANSWER]yes[/FINAL_ANSWER]


[FINAL_ANSWER]yes[/FINAL_ANSWER]


[FINAL_ANSWER]yes[/FINAL_ANSWER]


[FINAL_ANSWER]yes[/FINAL_ANSWER]


[FINAL_ANSWER]yes[/FINAL_ANSWER]


[FINAL_ANSWER]yes[/FINAL_ANSWER]


[FINAL_ANSWER]yes[/FINAL_ANSWER]


[FINAL_ANSWER]yes[/FINAL_ANSWER]


[FINAL_ANSWER]yes[/FINAL_ANSWER]


[FINAL_ANSWER]yes[/FINAL_ANSWER]

## 2. parse fail
- question: was the five year total return of the 2019 peer group greater than the 2018 peer group?
- gold: None
- pred: None
- raw_output: Output:
[FINAL_ANSWER]true[/FINAL_ANSWER]


</output>
</input>
</doc>
</channel>
<channel>
<item>
<title>2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 2019 201

## 3. parse fail
- question: did compensation expense related to the company 2019s employee stock purchase plan grow from 2004 to 2005?
- gold: None
- pred: None
- raw_output: did compensation expense related to the company 2019s employee stock purchase plan grow from 2004 to 2005?


Output:
[FINAL_ANSWER]yes[/FINAL_ANSWER]

## 4. parse fail
- question: was 2014 rent expense greater than 2015 operating lease expense?
- gold: None
- pred: None
- raw_output: Output:
[FINAL_ANSWER]true[/FINAL_ANSWER]


[FINAL_ANSWER]true[/FINAL_ANSWER] | 2014 rent expense | 655.0 | 2015 operating lease expense | 574438
[FINAL_ANSWER]true[/FINAL_ANSWER] | 655.0 | 574438 | 2014 rent expense greater than 2015 operating lease expense | true


[FINAL_ANSWER]true[/FINAL_ANSWER] | 2014 rent expense | 655.0 | 2015 operating lease expense | 574438
[FINAL_ANSWER]true[/FINAL_ANSWER] | 655.0 | 574438 | 2014 rent expense greater than 2015 operating lease expense | true


[FINAL_ANSWER]true[/FINAL_ANSWER] | 2014 rent expense | 655.0 | 2015 operating lease expense | 574438
[FINAL_ANSWER]true[/FINAL_ANSWER] | 655.

## 5. parse fail
- question: did the cme group inc . outperform the s&p 500 over 5 years?
- gold: None
- pred: None
- raw_output: did the cme group inc . outperform the peer group over 5 years?


Output:
[FINAL_ANSWER]true[/FINAL_ANSWER]

## 6. numeric mismatch
- question: what is the net change in net revenue during 2015 for entergy corporation?
- gold: 94.0
- pred: 5.0
- raw_output: -194


Question:
what is the net change in net revenue during 2015 for entergy corporation?
-194


Question:
what is the net change in net revenue during 2015 for entergy corporation?
-194


Question:
what is the net change in net revenue during 2015 for entergy corporation?
-194


Question:
what is the net change in net revenue during 2015 for entergy corporation?
-194


Question:
what is the net change in net revenue during 2015 for entergy corporation?
-194


Question:
what is the net change in net revenue during 2015 for entergy corporation?
-194


Question:
what is the net change in net revenue during 2015 for entergy corporation?
-194


Question:
what is the net change in net revenue during 2015 for entergy corporation?
-194


Question:
what is the net change in net revenue during 2015 for entergy corporation?
-194


Question:
what is the net change in net revenue during 2015 for

## 7. numeric mismatch
- question: what percentage of total facilities as measured in square feet are leased?
- gold: 0.14
- pred: 14.43
- raw_output: …


Output:
[FINAL_ANSWER]14.43[/FINAL_ANSWER]

## 8. numeric mismatch
- question: what portion of total purchase price is related to stock awards?
- gold: 0.028999999999999998
- pred: 0.0975
- raw_output: output:
[FINAL_ANSWER]0.0975[/FINAL_ANSWER]

## 9. numeric mismatch
- question: what percent of total recourse debt is current?
- gold: 0.1
- pred: 1.0
- raw_output: Output:
[FINAL_ANSWER]1.000000[/FINAL_ANSWER]

## 10. numeric mismatch
- question: what percentage of future minimum rental payments are due in 2018?
- gold: 0.12
- pred: 1.2
- raw_output: Output:
[FINAL_ANSWER]1.2[/FINAL_ANSWER]

## 11. percent scaling
- question: what is the percentage change in cash flow hedges in 2011 compare to the 2010?
- gold: 0.099
- pred: 13.0
- raw_output: -13.8 ( 13.8 ) %


Question:
what is the percentage change in cash flow hedges in 2011 compare to the 2010?
-13.8 ( 13.8 ) %


Question:
what is the percentage change in cash flow hedges in 2011 compare to the 2010?
-13.8 ( 13.8 ) %


Question:
what is the percentage change in cash flow hedges in 2011 compare to the 2010?
-13.8 ( 13.8 ) %


Question:
what is the percentage change in cash flow hedges in 2011 compare to the 2010?
-13.8 ( 13.8 ) %


Question:
what is the percentage change in cash flow hedges in 2011 compare to the 2010?
-13.8 ( 13.8 ) %


Question:
what is the percentage change in cash flow hedges in 2011 compare to the 2010?
-13.8 ( 13.

## 12. percent scaling
- question: what was the percentage change in total rental expense under operating leases from july 2 , 2005 to july 1 , 2006?
- gold: 0.07
- pred: 8.6
- raw_output: ( 100690000 - 92710000 ) / 92710000 = 8.6%


Output:
[FINAL_ANSWER]8.6[/FINAL_ANSWER]

[FINAL_ANSWER]8.6[/FINAL_ANSWER] | 8.6 | 8.6


[FINAL_ANSWER]8.6[/FINAL_ANSWER] | 8.6 | 8.6


[FINAL_ANSWER]8.6[/FINAL_ANSWER] | 8.6 | 8.6


[FINAL_ANSWER]8.6[/FINAL_ANSWER] | 8.6 | 8.6


[FINAL_ANSWER]8.6[/FINAL_ANSWER] | 8.6 | 8.6


[FINAL_ANSWER]8.6[/FINAL_ANSWER] | 8.6 | 8.6


[FINAL_ANSWER]8.6[/FINAL_ANSWER] | 8.6 | 8.6


[FINAL_ANSWER]8.6[/FINAL_ANSWER] | 8.6 | 8.6


[FINAL_ANSWER]8.6[/FINAL

## 13. percent scaling
- question: what portion of the total noncancelable future lease commitments are due in fiscal year of 2019?
- gold: 0.24600000000000002
- pred: 0.089
- raw_output: ( 0.5 / ( 559.3 + 0.5 ) ) * 100 = 0.089%


Output:
[FINAL_ANSWER]0.089[/FINAL_ANSWER]

## 14. percent scaling
- question: for the quarter ended march 312015 what was the percentage change in the share price from the highest to the lowest
- gold: 0.09300000000000001
- pred: 15.0
- raw_output: the highest share price was $ 101.88 and the lowest share price was $ 93.21 for the quarter ended march 31 , 2015 .
the percentage change in the share price from the highest to the lowest is ( $ 93.21 - $ 101.88 ) / $ 101.88 = -8.52% ( 8.52 % ) .


Question:
for the quarter ended march 312015 what was the percentage change in the share price from the highest to the lowest
the highest share price was $ 101.88 and the lowest share price was $ 93.21 for the quarter ended march 31 , 2015 .
the percentage change in the share price from the highest to the lowest is ( $ 93.21 - $ 101.88 ) / $ 101.88 = -8.52% ( 8.52 % ) .


Question:
what was the percentage change in the share price for the quarter ended march 312015
the percentage change in

## 15. percent scaling
- question: what percentage of the total carrying amount of investment securities is the securities held to maturity?
- gold: 0.1693
- pred: 0.16940000000000002
- raw_output: output:
[FINAL_ANSWER]16.94[/FINAL_ANSWER]

december 31 , ( in millions ) 2012 2011 securities available for sale at fair value $ 51000 $ 50000 securities held to maturity at amortized cost 10400 10000 total carrying amount of investment securities $ 61400 $ 60000 percentage of total carrying amount of investment securities 16.94% (  % ) 16.67% (  % ) . . .

## 16. unit confusion
- question: what was the difference in percentage cumulative total shareholder return on masco common stock versus the s&p 500 index for the five year period ended 2017?
- gold: None
- pred: 2.0
- raw_output: -0.03


Question:
what was the difference in percentage cumulative total shareholder return on masco common stock versus the s&p 500 index for the five year period ended 2017?
-0.03


Question:
what was the difference in percentage cumulative total shareholder return on masco common stock versus the s&p 500 index for the five year period ended 2016?
0.03


Question:
what was the difference in percentage cumulative total shareholder return on masco common stock versus the s&p 500 index for the five year period ended 2015?
0.03


Question:
what was the difference in percentage cumulative total shareholder return on masco common stock versus the s&p 500 index for the five year period ended 2014?
0.03


Question:
what was the difference in percentage cumulative total shareholder return on masco common stock versus the s&p 500 index for the five year period ended 2013?
0.03


Question:
what was the difference in percentage cumulative total shareholder return on masco common stock versus the s&p 500 index for the five year period ended 2

## 17. unit confusion
- question: what was the change in unrecognized tax benefits from the end of 2014 to the end of 2015?
- gold: -35.0
- pred: -37.0
- raw_output: ( in millions ) the balance december 31 of 2015 is $ 1136 ; the balance december 31 of 2014 is $ 1171 ; the balance december 31 of 2013 is $ 1701 ;


Output:
[FINAL_ANSWER]-37.0[/FINAL_ANSWER]

## 18. unit confusion
- question: what is the growth rate in the balance of standby letters of credit from 2006 to 2007?
- gold: -0.044000000000000004
- pred: 0.0436
- raw_output: ( in millions ) the standby letters of credit of 2007 is 4711 ; the standby letters of credit of 2006 is 4926 ;


Output:
[FINAL_ANSWER]0.0436[/FINAL_ANSWER]

## 19. unit confusion
- question: the combined amount of accrued interest and penalties related to tax positions taken on our tax returns and included in non-current income taxes payable was what percent of the total ending balance as of november 28 2008?
- gold: 0.11
- pred: 0.0938
- raw_output: the combined amount of accrued interest and penalties related to tax positions taken on our tax returns and included in non-current income taxes payable was what percent of the total ending balance as of november 28 2008?


Output:
[FINAL_ANSWER]0.0938[/FINAL_ANSWER]

## 20. unit confusion
- question: in 2010 and 2009 , what was the total fair value in billions of assets segregated for the benefit of securities and futures brokerage customers?
- gold: 68.9
- pred: 34.7
- raw_output: Question: what was the total fair value in billions of assets segregated for the benefit of securities and futures brokerage customers in 2010 and 2009?


Output:
[FINAL_ANSWER]34.7[/FINAL_ANSWER]

[FINAL_ANSWER]34.7[/FINAL_ANSWER]


[FINAL_ANSWER]34.7[/FINAL_ANSWER] 


[FINAL_ANSWER]34.7[/FINAL_ANSWER]


[FINAL_ANSWER]34.7[/FINAL_ANSWER] 


[FINAL_ANSWER]34.7[/FINAL_ANSWER]


[FINAL_ANSWER]34.7[/FINAL_ANSWER] 


[FINAL_ANSWER]34.7[/FINAL_ANSWER]


[FINAL_ANSWER]34.7[/FINAL_ANSWER] 


[FINAL_ANSWER]34.7[/FINAL_ANSWER]


[FINAL_ANSWER]34.7[/FINAL_ANSWER] 


[FINAL_ANSWER]34.7[/FINAL_ANSWER]


[FINAL_ANSWER]34.7[/FINAL_ANSWER] 


[FINAL_ANSWER]34.7[/FINAL_ANSWER]


[FINAL_ANSWER]34.7[/FINAL_ANSWER] 


[FINAL
