"""Входные и выходные проверки GX с сохранением Data Docs."""

import great_expectations as gx
import pandas as pd

from source import DESTINATIONS

E = gx.expectations


def input_checks():
    return [
        E.ExpectColumnValuesToNotBeNull(column='patient_id'),
        E.ExpectColumnValuesToBeUnique(column='patient_id'),
        E.ExpectColumnValuesToBeBetween(column='age', min_value=0, max_value=120),
        E.ExpectColumnValuesToBeBetween(column='length_of_stay', min_value=1, max_value=365),
        E.ExpectColumnValuesToBeInSet(column='readmitted_30_days', value_set=['Yes', 'No']),
        E.ExpectColumnValuesToBeInSet(column='discharge_destination', value_set=sorted(DESTINATIONS)),
    ]


def output_checks(destinations):
    return [
        E.ExpectColumnValuesToNotBeNull(column='encounter_count'),
        E.ExpectColumnValuesToBeInSet(column='destination_id', value_set=destinations),
        E.ExpectCompoundColumnsToBeUnique(column_list=['destination_id', 'gender']),
        E.ExpectColumnValuesToBeBetween(column='readmission_rate', min_value=0, max_value=1),
        E.ExpectColumnPairValuesAToBeGreaterThanB(column_A='encounter_count', column_B='readmitted_count', or_equal=True),
        E.ExpectTableRowCountToBeBetween(min_value=1),
    ]


def validate(frame, scope, checks, docs_path):
    context = gx.get_context(mode='ephemeral')
    asset = context.data_sources.add_pandas(name='hospital').add_dataframe_asset(name=scope)
    batch = asset.add_batch_definition_whole_dataframe('batch')
    suite = context.suites.add(gx.ExpectationSuite(name=scope))
    for check in checks:
        suite.add_expectation(check)
    validation = context.validation_definitions.add(gx.ValidationDefinition(name=scope, data=batch, suite=suite))
    result = validation.run(batch_parameters={'dataframe': frame}, result_format={'result_format': 'SUMMARY', 'partial_unexpected_count': 0})
    context.add_data_docs_site('local', {
        'class_name': 'SiteBuilder',
        'store_backend': {'class_name': 'TupleFilesystemStoreBackend', 'base_directory': str(docs_path.resolve())},
        'site_index_builder': {'class_name': 'DefaultSiteIndexBuilder', 'show_cta_footer': False},
    })
    context.build_data_docs(site_names=['local'])
    return {
        'passed': int(result.statistics['successful_expectations']),
        'total': int(result.statistics['evaluated_expectations']),
        'failed': [item.expectation_config.type for item in result.results if not item.success],
    }


def numeric_input(frame):
    columns = ['age', 'length_of_stay']
    return frame.assign(**{name: pd.to_numeric(frame[name]) for name in columns})
