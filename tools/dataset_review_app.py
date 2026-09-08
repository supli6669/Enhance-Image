"""Local-only review UI. Run with --server.address 127.0.0.1; never deploy publicly."""
from pathlib import Path
import csv
import json
import os
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.review_dataset import FIELDS, digest


def read_decisions(directory):
    record = json.loads((directory / 'review.json').read_text(encoding='utf-8'))
    if digest(directory / 'review_candidates.csv') != record['candidates_sha256']:
        raise ValueError('Original review evidence changed')
    with (directory / 'decisions.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    return record, rows


def save_decision(directory, row_id, decision, reviewer, expected_hash):
    path = directory / 'decisions.csv'
    if not reviewer.strip():
        raise ValueError('Nhập tên người duyệt trước khi lưu')
    if digest(path) != expected_hash:
        raise ValueError('Quyết định đã thay đổi ở phiên khác. Tải lại trang trước khi lưu.')
    _, rows = read_decisions(directory)
    row = next(r for r in rows if r['id'] == row_id)
    allowed = {'pending', 'distinct', 'same_group', 'exclude_left', 'exclude_right', 'exclude_both'} if row['right'] else {'pending', 'keep', 'exclude'}
    if decision not in allowed:
        raise ValueError('Invalid review decision')
    row.update(decision=decision, reviewer=reviewer.strip())
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    st.set_page_config(page_title='Duyệt dataset', layout='wide')
    st.title('Duyệt dataset trước training')
    st.caption('Chạy cục bộ. Điểm tương đồng là gợi ý; không xác nhận danh tính. Không tự phê duyệt split.')
    reports = Path(os.environ.get('DATASET_REVIEW_ROOT', str(ROOT / 'benchmarks/reports')))
    directories = sorted(p.parent for p in reports.glob('*/review.json'))
    if not directories:
        st.info('Chưa có bộ review hoàn chỉnh. Chờ quét xong rồi tải lại trang.')
        for scan in sorted(reports.glob('*/scan.json')):
            data = json.loads(scan.read_text(encoding='utf-8'))
            total = len(data['entries'])
            done = len(list((scan.parent / 'features').glob('*.json')))
            st.progress(min(done / max(total, 1), 1), text=f'{scan.parent.name}: {done}/{total} ảnh đã lưu đặc trưng')
        if st.button('Tải lại tiến độ'):
            st.rerun()
        return
    directory = st.selectbox('Bộ review', directories, format_func=lambda p: p.name)
    try:
        record, rows = read_decisions(directory)
    except (ValueError, OSError) as error:
        st.error(str(error))
        return
    decision_hash = digest(directory / 'decisions.csv')
    pending = sum(r['decision'] == 'pending' for r in rows)
    st.metric('Mục còn chờ duyệt', pending)
    reviewer = st.text_input('Người duyệt')
    only_pending = st.checkbox('Chỉ hiện mục chưa duyệt', value=True)
    cross_only = st.checkbox('Chỉ hiện cặp nằm ở hai tập khác nhau')
    reason = st.selectbox('Loại gợi ý', ['Tất cả'] + sorted({r['reason'] for r in rows}))
    visible = [r for r in rows if (not only_pending or r['decision'] == 'pending')
               and (not cross_only or (r['right'] and r['left_split'] != r['right_split']))
               and (reason == 'Tất cả' or r['reason'] == reason)]
    if not visible:
        st.info('Không có mục phù hợp bộ lọc. Tắt bộ lọc để xem lại quyết định.')
        return
    row = st.selectbox('Mục cần duyệt', visible, format_func=lambda r: f"{r['id']} · {r['reason']} · {r['left']}")
    st.write(f"{row['left_split']} / {row['right_split']} — Điểm gợi ý: {row['score'] or 'không có'}")
    indices = {entry['path']: i for i, entry in enumerate(record['entries'])}
    columns = st.columns(2)
    for column, key in zip(columns, ('left', 'right')):
        if row[key]:
            column.image(str(directory / f'thumbnails/{indices[row[key]]:06d}.jpg'), caption=row[key], width=320)
    choices = ['pending', 'distinct', 'same_group', 'exclude_left', 'exclude_right', 'exclude_both'] if row['right'] else ['pending', 'keep', 'exclude']
    st.caption('same_group: cùng nhóm người/ảnh; khi tạo split sẽ giữ holdout trước, rồi validation. exclude: bỏ khỏi split mới, không xóa ảnh gốc. Nếu ảnh không rõ, giữ pending.')
    decision = st.selectbox('Quyết định', choices, index=choices.index(row['decision']), key=f'decision_{directory.name}_{row["id"]}')
    if st.button('Lưu quyết định'):
        try:
            save_decision(directory, row['id'], decision, reviewer, decision_hash)
        except ValueError as error:
            st.error(str(error))
        else:
            st.rerun()


if __name__ == '__main__':
    main()
