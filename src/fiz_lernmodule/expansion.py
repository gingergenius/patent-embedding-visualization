# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.cloud import storage
from pandas.io import gbq
import pandas as pd
import pickle
import re
import os

class PatentLandscapeExpander:
    """Class for L1&L2 expansion as 'Automated Patent Landscaping' describes.

    This object takes a seed set and a Google Cloud BigQuery project name and
    exposes methods for doing expansion of the project. The logical entry-point
    to the class is load_from_disk_or_do_expansion, which checks for cached
    expansions for the given self.seed_name, and if a previous run is available
    it will load it from disk and return it; otherwise, it does L1 and L2
    expansions, persists it in a cached 'data/[self.seed_name]/' directory,
    and returns the data to the caller.
    """
    seed_file = None
    # BigQuery must be enabled for this project
    bq_project = 'patent-landscape-165715'
    patent_dataset = 'patents-public-data:patents.publications_latest'
    #tmp_table = 'patents._tmp'
    l1_tmp_table = 'patents._l1_tmp'
    l2_tmp_table = 'patents._l2_tmp'
    antiseed_tmp_table = 'patents.antiseed_tmp'
    country_codes = set(['US'])
    num_anti_seed_patents = 15000
    us_only = True

    # ratios and multipler for finding uniquely common CPC codes from seed set
    min_ratio_of_code_to_seed = 0.04
    min_seed_multiplier = 50.0

    # persisted expansion information
    training_data_full_df = None
    seed_patents_df = None
    l1_patents_df = None
    l2_patents_df = None
    anti_seed_patents = None
    seed_data_path = None

    def __init__(self, seed_file, seed_name, bq_project=None, patent_dataset=None, num_antiseed=None, us_only=None, prepare_training=True):
        self.seed_file = seed_file
        self.seed_data_path = os.path.join('data', seed_name)

        if bq_project is not None:
            self.bq_project = bq_project
        if patent_dataset is not None:
            self.patent_dataset = patent_dataset
        #if tmp_table is not None:
        #    self.tmp_table = tmp_table
        if num_antiseed is not None:
            self.num_anti_seed_patents = num_antiseed
        if us_only is not None:
            self.us_only = us_only         

        self.prepare_training = prepare_training                   

    def load_seeds_from_bq(self, seed_df):
        where_clause = ",".join("'" + seed_df.PubNum + "'")

        if self.us_only:
            seed_patents_query = '''
            SELECT
            b.publication_number,
            'Seed' as ExpansionLevel,
            STRING_AGG(citations.publication_number) AS refs,
            STRING_AGG(cpcs.code) AS cpc_codes
            FROM
            `patents-public-data.patents.publications` AS b,
            UNNEST(citation) AS citations,
            UNNEST(cpc) AS cpcs
            WHERE
            REGEXP_EXTRACT(b.publication_number, r'\w+-(\w+)-\w+') IN
            (
            {}
            )
            AND b.country_code = 'US'
            AND citations.publication_number != ''
            AND cpcs.code != ''
            GROUP BY b.publication_number
            ;
            '''.format(where_clause)
        else:
            seed_patents_query = '''
            SELECT
            b.publication_number,
            'Seed' as ExpansionLevel,
            STRING_AGG(citations.publication_number) AS refs,
            STRING_AGG(cpcs.code) AS cpc_codes
            FROM
            `patents-public-data.patents.publications` AS b,
            UNNEST(citation) AS citations,
            UNNEST(cpc) AS cpcs
            WHERE
            b.publication_number IN
            (
            {}
            )
            AND citations.publication_number != ''
            AND cpcs.code != ''
            GROUP BY b.publication_number
            ;
            '''.format(where_clause)

        seed_patents_df = gbq.read_gbq(
            query=seed_patents_query,
            project_id=self.bq_project,
            verbose=False,
            dialect='standard')

        return seed_patents_df

    def load_seed_pubs(self, seed_file=None):
        if seed_file is None:
            seed_file = self.seed_file

        #if self.us_only:
        seed_df = pd.read_csv(seed_file, header=None, names=['PubNum'], dtype={'PubNum': 'str'})
        #else:
        #    seed_df = pd.read_csv(seed_file, header=None, names=['publication_number'], dtype={'publication_number': 'str'})

        return seed_df

    def bq_get_num_total_patents(self):

        if self.us_only:
            num_patents_query = """
                SELECT
                COUNT(publication_number) AS num_patents
                FROM
                `patents-public-data.patents.publications` AS b
                WHERE
                country_code = 'US'
            """
        else:
            num_patents_query = """
                SELECT
                COUNT(publication_number) AS num_patents
                FROM
                `patents-public-data.patents.publications` AS b
            """
        
        num_patents_df = gbq.read_gbq(
            query=num_patents_query,
            project_id=self.bq_project,
            verbose=False,
            dialect='standard')
        return num_patents_df

    def get_cpc_counts(self, seed_publications=None):
        where_clause = '1=1'
        if seed_publications is not None:
            if self.us_only:
                where_clause = """
                REGEXP_EXTRACT(b.publication_number, r'\w+-(\w+)-\w+') IN
                    (
                    {}
                    )
                """.format(",".join("'" + seed_publications + "'"))
            else:
                where_clause = """
                b.publication_number IN
                    (
                    {}
                    )
                """.format(",".join("'" + seed_publications + "'"))


        if self.us_only:
            cpc_counts_query = """
                SELECT
                cpcs.code,
                COUNT(cpcs.code) AS cpc_count
                FROM
                `patents-public-data.patents.publications` AS b,
                UNNEST(cpc) AS cpcs
                WHERE
                {}
                AND cpcs.code != ''
                AND country_code = 'US'
                GROUP BY cpcs.code
                ORDER BY cpc_count DESC;
                """.format(where_clause)
        else:
            cpc_counts_query = """
                SELECT
                cpcs.code,
                COUNT(cpcs.code) AS cpc_count
                FROM
                `patents-public-data.patents.publications` AS b,
                UNNEST(cpc) AS cpcs
                WHERE
                {}
                AND cpcs.code != ''
                GROUP BY cpcs.code
                ORDER BY cpc_count DESC;
                """.format(where_clause)

        return gbq.read_gbq(
            query=cpc_counts_query,
            project_id=self.bq_project,
            verbose=False,
            dialect='standard')

    def compute_uniquely_common_cpc_codes_for_seed(self, seed_df):
        '''
        Queries for CPC counts across all US patents and all Seed patents, then finds the CPC codes
        that are 50x more common in the Seed set than the rest of the patent corpus (and also appear in
        at least 5% of Seed patents). This then returns a Pandas dataframe of uniquely common codes
        as well as the table of CPC counts for reference. Note that this function makes several
        BigQuery queries on multi-terabyte datasets, so expect it to take a couple minutes.
        
        You should call this method like:
        uniquely_common_cpc_codes, cpc_counts_df = \
            expander.compute_uniquely_common_cpc_codes_for_seed(seed_df)
            
        where seed_df is the result of calling load_seed_pubs() in this class.
        '''

        print('Querying for all US CPC Counts')
        us_cpc_counts_df = self.get_cpc_counts()
        print(us_cpc_counts_df.shape)
        print('Querying for Seed Set CPC Counts')
        seed_cpc_counts_df = self.get_cpc_counts(seed_df.PubNum)
        print(seed_cpc_counts_df.shape)
        print("Querying to find total number of US patents")
        num_patents_df = self.bq_get_num_total_patents()

        num_seed_patents = seed_df.count().values[0]
        num_us_patents = num_patents_df['num_patents'].values[0]

        # Merge/join the dataframes on CPC code, suffixing them as appropriate
        cpc_counts_df = us_cpc_counts_df.merge(
            seed_cpc_counts_df, on='code', suffixes=('_us', '_seed')) \
            .sort_values(ascending=False, by=['cpc_count_seed'])

        # For each CPC code, calculate the ratio of how often the code appears
        #  in the seed set vs the number of total seed patents
        cpc_counts_df['cpc_count_to_num_seeds_ratio'] = cpc_counts_df.cpc_count_seed / num_seed_patents
        # Similarly, calculate the ratio of CPC document frequencies vs total number of US patents
        cpc_counts_df['cpc_count_to_num_us_ratio'] = cpc_counts_df.cpc_count_us / num_us_patents
        # Calculate how much more frequently a CPC code occurs in the seed set vs full corpus of US patents
        cpc_counts_df['seed_relative_freq_ratio'] = \
            cpc_counts_df.cpc_count_to_num_seeds_ratio / cpc_counts_df.cpc_count_to_num_us_ratio

        # We only care about codes that occur at least ~4% of the time in the seed set
        # AND are 50x more common in the seed set than the full corpus of US patents
        uniquely_common_cpc_codes = cpc_counts_df[
            (cpc_counts_df.cpc_count_to_num_seeds_ratio >= self.min_ratio_of_code_to_seed)
            &
            (cpc_counts_df.seed_relative_freq_ratio >= self.min_seed_multiplier)]

        return uniquely_common_cpc_codes, cpc_counts_df


    def get_set_of_refs_filtered_by_country(self, seed_refs_series, country_codes):
        '''
        Uses the refs column of the BigQuery on the seed set to compute the set of
        unique references out of the Seed set.
        '''

        all_relevant_refs = set()
        for refs in seed_refs_series:
            for ref in refs.split(','):
                if self.us_only:
                    country_code = re.sub(r'(\w+)-(\w+)-\w+', r'\1', ref)
                    if country_code in country_codes:
                        all_relevant_refs.add(ref)
                else:
                    all_relevant_refs.add(ref)

        return all_relevant_refs


    # Expansion Functions
    def load_df_to_bq_tmp(self, df, tmp_table):
        '''
        This function inserts the provided dataframe into a temp table in BigQuery, which
        is used in other parts of this class (e.g. L1 and L2 expansions) to join on by
        patent number.
        '''
        print('Loading dataframe with cols {}, shape {}, to {}'.format(
            df.columns, df.shape, tmp_table))
        gbq.to_gbq(
            dataframe=df,
            destination_table=tmp_table,
            project_id=self.bq_project,
            if_exists='replace',
            verbose=False)

        print('Completed loading temp table.')

    def expand_l2(self, refs_series):

        if self.us_only:
            self.load_df_to_bq_tmp(pd.DataFrame(refs_series, columns=['pub_num']), self.l2_tmp_table)

            expansion_query = '''
                SELECT
                b.publication_number,
                'L2' AS ExpansionLevel,
                STRING_AGG(citations.publication_number) AS refs
                FROM
                `patents-public-data.patents.publications` AS b,
                `{}` as tmp,
                UNNEST(citation) AS citations
                WHERE
                (
                    b.publication_number = tmp.pub_num
                )
                AND citations.publication_number != ''
                GROUP BY b.publication_number
                ;
            '''.format(self.l2_tmp_table)
        else:
            self.load_df_to_bq_tmp(pd.DataFrame(refs_series, columns=['publication_number']), self.l2_tmp_table)

            expansion_query = '''
                SELECT
                b.publication_number,
                'L2' AS ExpansionLevel,
                STRING_AGG(citations.publication_number) AS refs
                FROM
                `patents-public-data.patents.publications` AS b,
                `{}` as tmp,
                UNNEST(citation) AS citations
                WHERE
                (
                    b.publication_number = tmp.publication_number
                )
                AND citations.publication_number != ''
                GROUP BY b.publication_number
                ;
            '''.format(self.l2_tmp_table)
            
        #print(expansion_query)
        expansion_df = gbq.read_gbq(
            query=expansion_query,
            project_id=self.bq_project,
            verbose=False,
            dialect='standard')

        return expansion_df

    def expand_l1(self, cpc_codes_series, refs_series):
        if self.us_only:
            self.load_df_to_bq_tmp(pd.DataFrame(refs_series, columns=['pub_num']), self.l1_tmp_table)
        else:
            self.load_df_to_bq_tmp(pd.DataFrame(refs_series, columns=['publication_number']), self.l1_tmp_table)

        cpc_where_clause = ",".join("'" + cpc_codes_series + "'")

        if self.us_only:
            expansion_query = '''
                SELECT DISTINCT publication_number, ExpansionLevel, refs
                FROM
                (
                SELECT
                b.publication_number,
                'L1' as ExpansionLevel,
                STRING_AGG(citations.publication_number) AS refs
                FROM
                `patents-public-data.patents.publications` AS b,
                UNNEST(citation) AS citations,
                UNNEST(cpc) AS cpcs
                WHERE
                (
                    cpcs.code IN
                    (
                    {}
                    )
                )
                AND citations.publication_number != ''
                AND country_code IN ('US')
                GROUP BY b.publication_number

                UNION ALL

                SELECT
                b.publication_number,
                'L1' as ExpansionLevel,
                STRING_AGG(citations.publication_number) AS refs
                FROM
                `patents-public-data.patents.publications` AS b,
                `{}` as tmp,
                UNNEST(citation) AS citations
                WHERE
                (
                    b.publication_number = tmp.pub_num
                )
                AND citations.publication_number != ''
                GROUP BY b.publication_number
                )
                ;
            '''.format(cpc_where_clause, self.l1_tmp_table)
        else:
            expansion_query = '''
                SELECT DISTINCT publication_number, ExpansionLevel, refs
                FROM
                (
                SELECT
                b.publication_number,
                'L1' as ExpansionLevel,
                STRING_AGG(citations.publication_number) AS refs
                FROM
                `patents-public-data.patents.publications` AS b,
                UNNEST(citation) AS citations,
                UNNEST(cpc) AS cpcs
                WHERE
                (
                    cpcs.code IN
                    (
                    {}
                    )
                )
                AND citations.publication_number != ''
                GROUP BY b.publication_number

                UNION ALL

                SELECT
                b.publication_number,
                'L1' as ExpansionLevel,
                STRING_AGG(citations.publication_number) AS refs
                FROM
                `patents-public-data.patents.publications` AS b,
                `{}` as tmp,
                UNNEST(citation) AS citations
                WHERE
                (
                    b.publication_number = tmp.publication_number
                )
                AND citations.publication_number != ''
                GROUP BY b.publication_number
                )
                ;
            '''.format(cpc_where_clause, self.l1_tmp_table)

        #print(expansion_query)
        expansion_df = gbq.read_gbq(
            query=expansion_query,
            project_id=self.bq_project,
            verbose=False,
            dialect='standard')

        return expansion_df

    def anti_seed(self, seed_expansion_series):
        if self.us_only:
            self.load_df_to_bq_tmp(pd.DataFrame(seed_expansion_series, columns=['pub_num']), self.antiseed_tmp_table)
            anti_seed_query = '''
                SELECT DISTINCT
                b.publication_number,
                'AntiSeed' AS ExpansionLevel,
                rand() as random_num
                FROM
                `patents-public-data.patents.publications` AS b
                LEFT OUTER JOIN `{}` AS tmp ON b.publication_number = tmp.pub_num
                WHERE
                tmp.pub_num IS NULL
                AND country_code = 'US'
                ORDER BY random_num
                LIMIT {}
                # TODO: randomize results
                ;
            '''.format(self.antiseed_tmp_table, self.num_anti_seed_patents)
        else:
            self.load_df_to_bq_tmp(pd.DataFrame(seed_expansion_series, columns=['publication_number']), self.antiseed_tmp_table)
            anti_seed_query = '''
                SELECT DISTINCT
                b.publication_number,
                'AntiSeed' AS ExpansionLevel,
                rand() as random_num
                FROM
                `patents-public-data.patents.publications` AS b
                LEFT OUTER JOIN `{}` AS tmp ON b.publication_number = tmp.publication_number
                WHERE
                tmp.publication_number IS NULL
                ORDER BY random_num
                LIMIT {}
                # TODO: randomize results
                ;
            '''.format(self.antiseed_tmp_table, self.num_anti_seed_patents)

        #print('Anti-seed query:\n{}'.format(anti_seed_query))
        anti_seed_df = gbq.read_gbq(
            query=anti_seed_query,
            project_id=self.bq_project,
            verbose=False,
            dialect='standard')

        return anti_seed_df

    def load_training_data_from_pubs(self, training_publications_df):
        tmp_table = 'patents._tmp_training'
        self.load_df_to_bq_tmp(df=training_publications_df, tmp_table=tmp_table)

        if self.us_only:            
            training_data_query = '''
                SELECT DISTINCT
                    REGEXP_EXTRACT(LOWER(p.publication_number), r'\w+-(\w+)-\w+') as pub_num,
                    p.publication_number,
                    p.country_code,
                    p.family_id,
                    p.priority_date,
                    title.text as title_text,
                    abstract.text as abstract_text,
                    claims.text as claims_text,
                    STRING_AGG(DISTINCT citations.publication_number) AS refs,
                    STRING_AGG(DISTINCT cpcs.code) AS cpcs,
                    STRING_AGG(DISTINCT ipcs.code) AS ipcs,
                    STRING_AGG(DISTINCT assignees_harmonized.name) as assignees_harmonized
                FROM
                `patents-public-data.patents.publications` p,
                UNNEST(p.citation) AS citations,
                `{}` as tmp,
                UNNEST(p.title_localized) AS title,
                UNNEST(p.abstract_localized) AS abstract,
                UNNEST(p.claims_localized) AS claims,
                UNNEST(assignee_harmonized) AS assignees_harmonized,
                UNNEST(cpc) AS cpcs,
                UNNEST(ipc) AS ipcs
                WHERE
                    p.publication_number = tmp.publication_number
                    AND p.country_code = 'US' 
                    AND title.language = 'en'
                    AND abstract.language = 'en'
                    AND claims.language = 'en'
                GROUP BY p.publication_number, p.country_code, p.family_id, p.priority_date, title.text,
                            abstract.text, claims.text  
                ;
            '''.format(tmp_table)
        else:
            training_data_query = '''
                SELECT DISTINCT
                    REGEXP_EXTRACT(LOWER(p.publication_number), r'\w+-(\w+)-\w+') as pub_num,
                    p.publication_number,
                    p.country_code,
                    p.family_id,
                    p.priority_date,
                    title.text as title_text,
                    abstract.text as abstract_text,
                    claims.text as claims_text,
                    STRING_AGG(DISTINCT citations.publication_number) AS refs,
                    STRING_AGG(DISTINCT cpcs.code) AS cpcs,
                    STRING_AGG(DISTINCT ipcs.code) AS ipcs,
                    STRING_AGG(DISTINCT assignees_harmonized.name) as assignees_harmonized
                FROM
                `patents-public-data.patents.publications` p
                LEFT JOIN UNNEST(p.claims_localized) AS claims,
                UNNEST(p.citation) AS citations,
                UNNEST(cpc) AS cpcs,
                `{}` as tmp,
                UNNEST(p.title_localized) AS title,
                UNNEST(p.abstract_localized) AS abstract,                
                UNNEST(assignee_harmonized) AS assignees_harmonized,                
                UNNEST(ipc) AS ipcs
                WHERE
                    p.publication_number = tmp.publication_number
                    AND title.language = 'en'
                    AND abstract.language = 'en'
                GROUP BY p.publication_number, p.country_code, p.family_id, p.priority_date, title.text,
                            abstract.text, claims.text  
                ;
            '''.format(tmp_table)

        print('Loading patent texts from provided publication numbers.')
        #print('Training data query:\n{}'.format(training_data_query))
        training_data_df = gbq.read_gbq(
            query=training_data_query,
            project_id=self.bq_project,
            verbose=False,
            dialect='standard',
            configuration = {'query': {'useQueryCache': True, 'allowLargeResults': False}})

        print(training_data_df.shape)

        return training_data_df

    def do_full_expansion(self):
        '''
        Does a full expansion on seed set as described in paper, using seed set
        to derive an anti-seed for use in supervised learning stage.
        
        Call this method like:
        seed_patents_df, l1_patents_df, l2_patents_df, anti_seed_patents = \
            expander.do_full_expansion(seed_file)
        '''
        seed_df = self.load_seed_pubs(self.seed_file)
        print("Loaded {} seed publication numbers".format(len(seed_df)))

        seed_patents_df = self.load_seeds_from_bq(seed_df)
        print("Loaded {} seed patents from BigQuery".format(len(seed_patents_df)))

        if (self.prepare_training):
            # Level 1 Expansion
            ## getting unique seed CPC codes
            uniquely_common_cpc_codes, cpc_counts_df = \
                self.compute_uniquely_common_cpc_codes_for_seed(seed_df)
            ## getting all the references out of the seed set
            all_relevant_refs = self.get_set_of_refs_filtered_by_country(
                seed_patents_df.refs, self.country_codes)
            print('Got {} relevant seed refs'.format(len(all_relevant_refs)))

            if (len(all_relevant_refs) > 0):
                ## actually doing expansion with CPC and references
                l1_patents_df = self.expand_l1(
                    uniquely_common_cpc_codes.code, pd.Series(list(all_relevant_refs)))
                print('Shape of L1 expansion: {}'.format(l1_patents_df.shape))

                # Level 2 Expansion
                l2_refs = self.get_set_of_refs_filtered_by_country(
                    l1_patents_df.refs, self.country_codes)
                print('Got {} relevant L1->L2 refs'.format(len(l2_refs)))
                l2_patents_df = self.expand_l2(pd.Series(list(l2_refs)))
                print('Shape of L2 expansion: {}'.format(l2_patents_df.shape)) 
            else:
                l1_patents_df = pd.DataFrame(columns=["publication_number"])
                l2_patents_df = pd.DataFrame(columns=["publication_number"])

            # Get all publication numbers from Seed, L1, and L2
            ## for use in getting anti-seed
            all_pub_nums = pd.Series(seed_patents_df.publication_number) \
                .append(l1_patents_df.publication_number) \
                .append(l2_patents_df.publication_number)
            seed_and_expansion_pub_nums = set()
            for pub_num in all_pub_nums:
                seed_and_expansion_pub_nums.add(pub_num)
            print('Size of union of [Seed, L1, and L2]: {}'.format(len(seed_and_expansion_pub_nums)))

            # get the anti-seed set!
            anti_seed_df = self.anti_seed(pd.Series(list(seed_and_expansion_pub_nums)))
        else:
            l1_patents_df = pd.DataFrame(columns=["publication_number"])
            l2_patents_df = pd.DataFrame(columns=["publication_number"])
            anti_seed_df = pd.DataFrame(columns=["publication_number"])

        return seed_patents_df, l1_patents_df, l2_patents_df, anti_seed_df


    def derive_training_data_from_seeds(self):
        '''
        '''
        seed_patents_df, l1_patents_df, l2_patents_df, anti_seed_patents = self.do_full_expansion()

        if self.prepare_training:
            training_publications_df = seed_patents_df.append(anti_seed_patents)[['publication_number', 'ExpansionLevel']]
        else:
            training_publications_df = seed_patents_df[['publication_number', 'ExpansionLevel']]

        print('Loading training data text from {} publication numbers'.format(training_publications_df.shape))
        training_data_df = self.load_training_data_from_pubs(training_publications_df[['publication_number']])

        print('Merging labels into training data.')
        training_data_full_df = training_data_df.merge(training_publications_df, on=['publication_number'])

        return training_data_full_df, seed_patents_df, l1_patents_df, l2_patents_df, anti_seed_patents

    def load_from_disk_or_do_expansion(self):
        """Loads data for seed from disk, else derives/persists, then returns it.

        Checks for cached expansions for the given self.seed_name, and if a
        previous run is available it will load it from disk and return it;
        otherwise, it does L1 and L2 expansions, persists it in a cached
        'data/[self.seed_name]/' directory, and returns the data to the caller.
        """

        landscape_data_path = os.path.join(self.seed_data_path, 'landscape_data.pkl')

        if not os.path.exists(landscape_data_path):
            if not os.path.exists(self.seed_data_path):
                os.makedirs(self.seed_data_path)

            print('Loading landscape data from BigQuery.')
            training_data_full_df, seed_patents_df, l1_patents_df, l2_patents_df, anti_seed_patents = \
                self.derive_training_data_from_seeds()

            print('Saving landscape data to {}.'.format(landscape_data_path))
            with open(landscape_data_path, 'wb') as outfile:
                pickle.dump(
                    (training_data_full_df, seed_patents_df, l1_patents_df, l2_patents_df, anti_seed_patents),
                    outfile, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print('Loading landscape data from filesystem at {}'.format(landscape_data_path))
            with open(landscape_data_path, 'rb') as infile:

                landscape_data_deserialized = pickle.load(infile)

                training_data_full_df, seed_patents_df, l1_patents_df, l2_patents_df, anti_seed_patents = \
                    landscape_data_deserialized

        self.training_data_full_df = training_data_full_df
        self.seed_patents_df = seed_patents_df
        self.l1_patents_df = l1_patents_df
        self.l2_patents_df = l2_patents_df
        self.anti_seed_patents = anti_seed_patents

        return training_data_full_df, seed_patents_df, l1_patents_df, l2_patents_df, anti_seed_patents

    def sample_for_inference(self, train_data_util, sample_frac=0.20):
        if self.l1_patents_df is None:
            raise ValueError('No patents loaded yet. Run expansion first (e.g., load_from_disc_or_do_expansion)')

        inference_data_path = os.path.join(self.seed_data_path, 'landscape_inference_data.pkl')

        if not os.path.exists(inference_data_path):
            print('Loading inference data from BigQuery.')
            subset_l1_pub_nums = self.l1_patents_df[['publication_number']].sample(frac=sample_frac).reset_index(drop=True)

            l1_texts = self.load_training_data_from_pubs(subset_l1_pub_nums)

            l1_subset = l1_texts[['publication_number', 'abstract_text', 'refs', 'cpcs']]

            # encode the data using the training data util
            padded_abstract_embeddings, refs_one_hot, cpc_one_hot = \
                train_data_util.prep_for_inference(l1_subset.abstract_text, l1_subset.refs, l1_subset.cpcs)

            print('Saving inference data to {}.'.format(inference_data_path))
            with open(inference_data_path, 'wb') as outfile:
                pickle.dump(
                    (subset_l1_pub_nums, l1_texts, padded_abstract_embeddings, refs_one_hot, cpc_one_hot),
                    outfile, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print('Loading inference data from filesystem at {}'.format(inference_data_path))
            with open(inference_data_path, 'rb') as infile:
                inference_data_deserialized = pickle.load(infile)

                subset_l1_pub_nums, l1_texts, padded_abstract_embeddings, refs_one_hot, cpc_one_hot = \
                    inference_data_deserialized

        return subset_l1_pub_nums, l1_texts, padded_abstract_embeddings, refs_one_hot, cpc_one_hot
