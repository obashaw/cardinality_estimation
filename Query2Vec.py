import numpy as np
import os
import re
import pandas as pd
import torch

class Query2Vec():
    def __init__(self, dataFile, columnInfoFile, n_clauses=3):
        self.local = os.getcwd()
        self.columnInfo = pd.read_csv(os.path.join(self.local, columnInfoFile))
        self.n_clauses = n_clauses
        self.operator_dict = {'=':1, '>':2, '<':3, '>=':4,'<=': 5, '!=':6, '<>':6}
        self.max_len = 0
        
    
    def parse(self, data):
        lines = data.split('\n')[:-1]
        self.setTableInfo()
        linesTemp = lines
        cards = [linesTemp[i].split('#')[-1:] for i in range(len(linesTemp))]
        c = [i[0] for i in cards]
        cardinalities = list(map(int, c))
        vectorized_lines = self.vectorizeQueries(lines)
        query_tensor = self.padQueries(vectorized_lines)
        return query_tensor, cardinalities

    def setTableInfo(self):
        ends = self.columnInfo.iloc[:,0].str.find('.').to_numpy()

        cols = self.columnInfo.iloc[:,0].to_numpy()
    
        cards = self.columnInfo.iloc[:,3].to_numpy()
        tables = np.array([cols[i][:ends[i]] for i in range(len(cols))])
        tableInfo= {tables[0]: cards[0]}
        for i in range(1, len(tables)):
            if not tables[i] in tableInfo.keys():
                tableInfo[tables[i]]=cards[i]
        self.tableInfo = tableInfo
        self.max_cardinality = np.max(cards)

        return

    # not used
    def annotateQueries(self, lines):
        clause_types = { 0: 'FROM', 1: 'JOIN', 2:'WHERE'}
        annotatedLines=[]
        for l in lines:
            splitIndicies = [j for j, x in enumerate(l) if x == '#']
            newLine = []
            curr = 0
            for i in range(len(splitIndicies)):
                if  l[splitIndicies[i]-1] != '#':
                    if i == 0:
                        newLine.append(clause_types[i]+' '+l[curr:splitIndicies[i]])
                    else:
                        newLine.append(clause_types[i]+l[curr:splitIndicies[i]])
                curr = splitIndicies[i]
            annotatedLines.append(" ".join(newLine))
        annotated = [" ".join(re.split('#|,', l)).upper() for l in annotatedLines]
        return annotated

    def vectorizeQueries(self, lines):
        clause_tokens = np.arange(self.n_clauses)
        annotatedLinesVect=[]
        for l in lines:
            splitIndicies = [j for j, x in enumerate(l) if x == '#']
            curr = 0
            vect3 = []
            for i in range(len(splitIndicies)):
                if  l[splitIndicies[i]-1] != '#':
                    clause = l[curr:splitIndicies[i]]
                    vect3.append(self.annotateClause(clause, i+1))
                curr = splitIndicies[i]
            vect3 = np.concatenate(vect3)
            if len(vect3) > self.max_len:
                self.max_len = len(vect3)
            annotatedLinesVect.append(vect3)
        return annotatedLinesVect


    def annotateClause(self, clause, clause_n):
        if clause_n ==1:
            tables = clause.split(',')

            table_feature_vect = [np.array([1,0,0])]
            for i in range(len(tables)):
                tname = tables[i].split(' ')[1]
                
                table_features = np.array([self.tableInfo[tname]/self.max_cardinality,0,0])
                table_feature_vect.append(table_features)
            return np.array(table_feature_vect)
        
        elif clause_n == 2:
            clause = clause[1:]
            joins = clause.split(',')

            join_feature_vect = [np.array([0,1,0])]
            for i in range(len(joins)):
                keys = joins[i].split('=')
             
                left_card = self.columnInfo.iloc[:,:][self.columnInfo.iloc[:,0]==keys[0]].iloc[0, 1:].to_numpy()[2]
                right_card = self.columnInfo.iloc[:,:][self.columnInfo.iloc[:,0]==keys[1]].iloc[0, 1:].to_numpy()[2]
                
                operator = self.operator_dict['=']
                feature_vect=np.array([left_card/self.max_cardinality, operator, right_card/self.max_cardinality])
                join_feature_vect.append(feature_vect)

            return np.array(join_feature_vect)
        elif clause_n == 3:
            clause = clause[1:]
            predicates = clause.split(',')
            pred_features_vect =  [np.array([0,0,1])]
            for i in range(0,len(predicates),3):
                pred = predicates[i:i+3]
                
                
                col_stats = self.columnInfo.iloc[:,:][self.columnInfo.iloc[:,0]==pred[0]].iloc[0, 1:]
                card = col_stats[2]
                unique_scaled = col_stats[3]/card
                pred_features = np.array([card/self.max_cardinality, self.operator_dict[pred[1]], unique_scaled])
                pred_features_vect.append(pred_features)
            return  np.array(pred_features_vect)
        else: 
            print('More than 3 clauses')
            return -1
    def padQueries(self, query_vectors):
        query_tensors = []
        for qv in query_vectors:
            qt = torch.FloatTensor(qv)
            pad = self.max_len - qt.size(0)
            z = torch.zeros(pad,qt.size(1))

            qt_padded = torch.cat((qt, z))
            query_tensors.append(qt_padded)
        
        return torch.stack(query_tensors)
