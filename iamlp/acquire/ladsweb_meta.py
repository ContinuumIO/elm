# TODO fill this out with functions that
# can validate whether a given product_name or product_number
# are valid, and if so, what is the file type expected on FTP.


def validate_product_number(product_number, context):
    if not isinstance(product_number, int):
        raise LadswebConfigError("Expected product_number {} in data_sources:{} "
                               "to be an int".format(product_number, context))
    # TODO look up in a metadata json to see if num exists
    return True

def validate_product_name(product_number, product_name, context):
    if not product_name or not isinstance(product_name, str):
        raise LadswebConfigError('Expected product_name {} in data_sources {} '
                               'to be a string'.format(product_name, context))
    # TODO look up product_name in a json of metadata about sources
    return True

def validate_time(product_number, product_name, years, data_days):
    # TODO look at metadata about temporal coverage
    return True # or raise LadswebConfigError

def validate_ladsweb_datasource(data_source_dict, context):
    ''' Raises error or returns True'''
    product_name = data_source_dict.get('product_name')
    product_number = data_source_dict.get('product_number')
    years = data_source_dict.get('years')
    data_days = data_source_dict.get('data_days')
    validate_product_number(product_number, context)
    validate_ladsweb_datasource(product_number, product_name, context)
    validate_time(product_number, product_name, years, data_days)
    return True