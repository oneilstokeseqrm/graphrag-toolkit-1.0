# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tfidf_matcher as tm
from typing import List, Optional

def score_values(values:List[str], match_values:str, limit:Optional[int]=None, ngram_length=3):
        
        values_to_score = values.copy()
        
        max_num_values_to_score =  len(values_to_score)
        if limit:
            max_num_values_to_score = min(limit, max_num_values_to_score)

        while len(values_to_score) <= max_num_values_to_score:
            values_to_score.append('')

        scored_values = {}

        try:
            
            matcher_results = tm.matcher(match_values, values_to_score, max_num_values_to_score, ngram_length)

            max_i = len(matcher_results.columns)
        
            for row_index in range(0, len(match_values)):
                for col_index in range(1, max_i, 3) :
                    value = matcher_results.iloc[row_index, col_index]
                    score = matcher_results.iloc[row_index, col_index+1]
                    if value not in scored_values:
                        scored_values[value] = score
                    else:
                        scored_values[value] = max(scored_values[value], score)
        except ValueError:
            scored_values = {v: 0.0 for v in values_to_score if v}
        
                
        sorted_scored_values = dict(sorted(scored_values.items(), key=lambda item: item[1], reverse=True))
        
        return sorted_scored_values