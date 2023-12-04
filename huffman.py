import numpy as np
from filecmp import cmp

## thuật toán huffman
class node:
    def __init__(self, count, index, name=""):
        self.count = float(count)
        self.index = index
        self.name = name
        if self.name == "": self.name = index
        self.word = ""
        self.isinternal = 0

    def __cmp__(self, other):
        return cmp(self.count, other.count)

    def report(self):
        if (self.index == 1):
            print('#Symbol\tCount\tCodeword')
        print('%s\t(%2.2g)\t%s' % (self.name, self.count, self.word))
        pass

    def associate(self, internalnode):
        self.internalnode = internalnode
        internalnode.leaf = 1
        internalnode.name = self.name
        pass


class internalnode:
    def __init__(self):
        self.leaf = 0
        self.child = []
        pass

    def children(self, child0, child1):
        self.leaf = 0
        self.child.append(child0)
        self.child.append(child1)
        pass


def find(f, seq):
    for item in seq:
        if f(item):
            return item


def iterate(c):
    if (len(c) > 1):

        ## sắp xếp các nút theo số lượng, sử dụng hàm __cmp__ được xác định trong lớp nút
        deletednode = c[0]  # tạo bản sao của nút nhỏ nhất
        second = c[1].index  ## chỉ mục của nút thứ hai nhỏ nhất

        # cộng  hai thành phần dưới cùng
        c[1].count += c[0].count
        del c[0]

        root = iterate(c)

        ## Điền thông tin mới vào danh sách
        ## tìm từ mã đã được tách / nối
        co = find(lambda p: p.index == second, c)
        deletednode.word = co.word + '0'
        c.append(deletednode)
        co.word += '1'
        co.count -= deletednode.count  ## khôi phục số node lượng chính xác

        ## tạo ra các nhánh mới trong cây
        newnode0 = internalnode()
        newnode1 = internalnode()
        treenode = co.internalnode  # tìm nút có hai nút con
        treenode.children(newnode0, newnode1)
        deletednode.associate(newnode0)
        co.associate(newnode1)
        pass
    else:
        c[0].word = ""
        root = internalnode()
        c[0].associate(root)
        pass
    return root


def encode(sourcelist, code):
    answer = ""
    for s in sourcelist:
        co = find(lambda p: p.name == s, code)

        answer = answer + co.word

    return answer


def decode(string, root):
    answer = []
    clist = list(string)
    # duyệt bắt đầu từ gốc
    currentnode = root
    for c in clist:
        if (c == '\n'):  continue  # trường hợp đặc biệt cho các ký tự mới
        assert (c == '0') or (c == '1')
        currentnode = currentnode.child[int(c)]
        if currentnode.leaf != 0:
            answer.append(str(currentnode.name))
            currentnode = root
        pass
    assert (currentnode == root)
    return answer


def makenodes(probs):
    m = 0
    c = []
    for p in probs:
        m += 1
        c.append(node(p[1], m, p[0]))
        pass
    return c