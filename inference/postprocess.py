
CLASS_NAMES = ['NUL', 'k_bank_name', 'v_bank_name', 'k_bank_branch_name', 'v_bank_branch_name', 'k_account_number',
               'v_account_number', 'k_account_type', 'v_account_type', 'k_account_name', 'v_account_name',
               'k_account_name_kana', 'v_account_name_kana', 'k_branch', 'v_branch', 'k_financial_institution',
               'v_financial_institution']


def post_process_kv(values):
    results = {}
    for idx, v in enumerate(values):
        if idx % 2 == 1 and idx > 1:
            field_name = CLASS_NAMES[idx-1][2:] if len(CLASS_NAMES) > idx - 1 else str(idx - 1)
            value = v[0]
            results[field_name] = value
    return results
