"""Local-only review UI. Run with --server.address 127.0.0.1; never deploy publicly."""
from pathlib import Path
import csv
import json
import hashlib
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
    experiments = sorted((reports.parent / 'splits').glob('*/quality.json'))
    if experiments:
        mode = st.radio('Cách xem', ['Duyệt nhanh — split thử nghiệm', 'Review chi tiết cũ'], horizontal=True)
        if mode.startswith('Duyệt nhanh'):
            quick_review(experiments)
            return
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


def quick_review(experiments):
    quality_path = st.selectbox('Split thử nghiệm', experiments, format_func=lambda p: p.parent.name)
    directory = quality_path.parent
    report = json.loads(quality_path.read_text(encoding='utf-8'))
    split = json.loads((directory / 'split.json').read_text(encoding='utf-8'))
    if digest(quality_path) != split['mitigation']['quality_sha256']:
        st.error('Báo cáo kỹ thuật đã thay đổi; cần tạo split mới.')
        return
    a,b,c = st.columns(3)
    a.metric('Ảnh train', report['train_count'])
    b.metric('Ảnh validation', report['validation_count'])
    c.metric('Tạm loại khỏi split', report['quarantined_count'])
    st.success('Không cần bấm duyệt hàng nghìn mục. Split thử nghiệm đã được tạo bằng quy tắc bảo thủ.')
    st.info('Chưa chứng nhận tách theo người. Có thể dùng để đo baseline thử nghiệm; chưa đủ để tự phát hành model production.')
    st.caption('Ảnh gốc và holdout được giữ nguyên. Xem mẫu là tùy chọn, không phải 40 quyết định bắt buộc.')
    sample_tab, exceptions_tab, excluded_tab = st.tabs(['Ảnh mẫu', 'Ngoại lệ kỹ thuật', 'Danh sách tạm loại'])
    for tab, rows in [(sample_tab, report['sample']), (exceptions_tab, report['shortlist'])]:
        with tab:
            if not rows:
                st.write('Không có ngoại lệ kỹ thuật cần hiển thị.')
            columns = st.columns(4)
            for index, row in enumerate(rows):
                name = hashlib.sha256(row['path'].encode()).hexdigest()+'.jpg'
                columns[index % 4].image(str(directory/'thumbnails'/name),
                    caption=f"{row['split']} · {row['path']} · {', '.join(row['flags'])}", width=192)
    with exceptions_tab:
        st.caption(f"Hiển thị tối đa 12/{report['quality_flag_count']} ảnh có cờ kỹ thuật. Đây là gợi ý xem lại, không tự kết luận ảnh kém chất lượng.")
    with excluded_tab:
        st.dataframe([{'path':name,'reason':reason} for name,reason in split['mitigation']['excluded'].items()], hide_index=True)
    st.code(f'python tools/prepare_baseline.py --experimental --split-dir "{directory}" --output benchmarks/reports/experiment_baseline_v1 --execute')


if __name__ == '__main__':
    main()
