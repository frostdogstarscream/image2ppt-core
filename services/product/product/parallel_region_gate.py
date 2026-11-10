"""Surface erasure depends only on locally affected, proven replacements."""
def region_ready(region, records, accepted):
    by_id={r['region_id']:r for r in records}
    for dependency in region['dependencies']:
        if dependency not in by_id or not set(by_id[dependency]['member_ids'])<=accepted:
            return False
    return True
