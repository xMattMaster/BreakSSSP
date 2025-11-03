"""
Data structure for efficient frontier management in BreakSSSP algorithm.

This module implements the specialized data structure described in Lemma 3.3 of the paper,
which supports efficient Insert, BatchPrepend, and Pull operations on key-value pairs with
bounded values.

The structure consists of:
    - D0: Block sequence for batch-prepended elements
    - D1: Block sequence for individually inserted elements, with BST index
    - Blocks: Linked lists containing up to M items

Key operations achieve amortized times:
    - Insert: O(max(1, M/B) * log(B/M))
    - BatchPrepend: O(max(1, M/B) * log(B/M))
    - Pull: O(M)

References:
    Lemma 3.3 in Duan et al. (2025)
"""

import functools
from typing import Optional, Tuple, List, Set, Dict

from bfprt import select_fast
from sortedcontainers import SortedKeyList


@functools.total_ordering
class _Item:
    """
    Item is the representation of a single node and its distance from the origin.

    It may be concatenated to other items in a linked list manner, forming a block. In that case,
    it also keeps track of the block that contains it.
    """

    __slots__ = ["key", "value", "prev", "next", "block"]

    def __init__(self, key:int, value: float):
        """
        Initialize an item with a key-value pair.

        :param key: the id of the node the item represents.
        :param value: the current distance from the origin.
        """
        self.key: int = key
        self.value: float = value
        self.prev: Optional["_Item"] = None
        self.next: Optional["_Item"] = None
        self.block: Optional["_Block"] = None

    def __eq__(self, other: "_Item"):
        """
        Check whether two items have the same distance from the origin.

        :param other: the other item to compare.
        :return: ``True`` if the two items have the same distance, ``False`` otherwise.
        """
        return self.value == other.value

    def __lt__(self, other: "_Item"):
        """
        Compare the distance of two nodes from the origin. Allows total ordering.

        :param other: the other item to compare.
        :return: ``True`` if the distance of the first item is less than the second item,
        ``False`` otherwise.
        """
        return self.value < other.value


class _Block:
    """
    Block is a linked list containing items.

    It may be concatenated to other blocks in a linked list manner, forming a block structure in
    a NodesStructure. In that case, can contain up to ``M`` items, after which the block will
    partition itself in two smaller blocks, containing half of the items each. It may also have
    an upper bound, limiting the maximum distance an item in it can have.

    """

    __slots__ = ("id", "ub", "head", "tail", "size", "prev", "next")

    def __init__(self, ub: Optional[float] = None):
        """
        Initialize a block, optionally with an upper bound.

        :param ub: the upper bound of the block (``None`` by default).
        """
        self.ub: Optional[float] = ub
        self.head: Optional[_Item] = None
        self.tail: Optional[_Item] = None
        self.size: int = 0
        self.prev: Optional["_Block"] = None
        self.next: Optional["_Block"] = None

    def append(self, item: _Item):
        """
        Append an item in the block.

        :param item: the item to append.
        """
        if self.ub is not None:
            assert item.value <= self.ub, "Item value is bigger than block UB"
        item.prev = self.tail
        item.next = None
        if self.tail is not None:
            self.tail.next = item
        else:
            self.head = item
        self.tail = item
        item.block = self
        self.size += 1

    def remove(self, item: _Item) -> int:
        """
        Remove an item from the block. It does not delete the item from memory.

        :param item: the item to remove.
        :return: the new size of the list.
        """
        if item.prev is not None:
            item.prev.next = item.next
        else:
            self.head = item.next
        if item.next is not None:
            item.next.prev = item.prev
        else:
            self.tail = item.prev
        item.prev = item.next = None
        item.block = None
        self.size -= 1
        return self.size

    def delete(self, item: _Item):
        """
        Remove an item from the block and delete it from memory.

        :param item: the item to delete.
        """
        self.remove(item)
        del item

    def split(self):
        """
        Split the block into smaller blocks.

        :return: the two smaller blocks.
        """
        items: List[_Item] = list(self)
        extra_items: List[_Item] = []
        median = items[select_fast(items, self.size // 2)].value
        if self.ub is not None:
            left = _Block(median)
            right = _Block(self.ub)
        else:
            left = _Block()
            right = _Block()
        for item in items:
            if item.value < median:
                left.append(item)
            elif item.value > median:
                right.append(item)
            else:
                extra_items.append(item)
        for item in extra_items:
            if right.size < self.size // 2 + self.size % 2:
                right.append(item)
            else:
                left.append(item)
        return left, right

    def is_empty(self):
        """
        Check if the block is empty.

        :return: ``True`` if the block is empty, ``False`` otherwise.
        """
        return self.size == 0

    def __iter__(self):
        """
        Iterate over the block, yielding its items.

        :return: the next item of the block if present, ``None`` otherwise.
        """
        current = self.head
        while current is not None:
            next_i = current.next
            yield current
            current = next_i


class _BlockSequence:
    """
    BlockSequence is a linked list containing blocks.

    BlockSequence is used for both D0 and D1. For the latter, a SortedKeyList, functioning as the
    BST, is also used.
    """

    __slots__ = ("head", "tail", "size")

    def __init__(self, initial_block: Optional[_Block] = None):
        """
        Initialize the BlockSequence, optionally with an initial block.

        :param initial_block: the block to initialize the BlockSequence with.
        """
        self.head: Optional[_Block] = initial_block
        self.tail: Optional[_Block] = initial_block
        self.size: int = 0 if initial_block is None else 1

    def delete(self, block: _Block):
        """
        Delete a block from the BlockSequence.

        :param block: the block to delete.
        """
        assert block.size == 0, "Cannot remove a non-empty block"
        if block.prev is not None:
            block.prev.next = block.next
        else:
            self.head = block.next
        if block.next is not None:
            block.next.prev = block.prev
        else:
            self.tail = block.prev
        block.prev = block.next = None
        del block
        self.size -= 1

    def substitute_block_with_split(self, orig: _Block, new: Tuple[_Block, _Block]):
        """
        Substitute a block in the BlockSequence with its split.

        :param orig: the original block to substitute.
        :param new: a tuple containing the two new blocks.
        """
        if orig.prev is not None:
            new[0].prev = orig.prev
            orig.prev.next = new[0]
        else:
            self.head = new[0]
        if orig.next is not None:
            new[1].next = orig.next
            orig.next.prev = new[1]
        else:
            self.tail = new[1]
        new[0].next = new[1]
        new[1].prev = new[0]
        orig.prev = orig.next = None
        del orig
        self.size += 1

    def batch_prepend(self, block: _Block, max_size: int):
        """
        Prepend a block to the BlockSequence.

        If the block is greater than ``max_size``, split it into a series of blocks such that the
        biggest block has at most size ``ceil(max_size / 2)``.

        :param block: the block to prepend.
        :param max_size: the maximum size the prepended block should have.
        """
        if block.size <= max_size:
            if self.head is not None:
                self.head.prev = block
            else:
                self.tail = block
            block.next = self.head
            block.prev = None
            self.head = block
            self.size += 1
            return
        blocks: List[_Block] = [block]
        processed_blocks: List[_Block] = []
        while blocks:
            current_block = blocks.pop()
            if current_block.size <= max_size // 2 + max_size % 2:
                processed_blocks.append(current_block)
                continue
            left, right = current_block.split()
            del current_block
            blocks.append(right)
            blocks.append(left)
        self.size += len(processed_blocks)
        for i in range(len(processed_blocks) - 1):
            processed_blocks[i].next = processed_blocks[i + 1]
            processed_blocks[i + 1].prev = processed_blocks[i]
        if self.head is not None:
            self.head.prev = processed_blocks[-1]
            processed_blocks[- 1].next = self.head
        else:
            self.tail = processed_blocks[-1]
            processed_blocks[-1].next = None
        self.head = processed_blocks[0]
        processed_blocks[0].prev = None
        block.prev = block.next = None
        del block

    def is_empty(self):
        """
        Check if the BlockSequence is empty.

        :return: ``True`` if the BlockSequence is empty, ``False`` otherwise.
        """
        return self.size == 0

    def __iter__(self):
        """
        Iterate over BlockSequence, yielding its items.

        :return: the next block of the sequence if present, ``None`` otherwise.
        """
        current = self.head
        while current is not None:
            next_b = current.next
            yield current
            current = next_b


class NodesStructure:
    """NodesStructure is the data structure defined in Lemma 3.3

    It is composed of two sequences of blocks, D0 and D1. D0 contains elements added via batch
    prepends, whilst D1 maintains elements from insert operations.
    """

    def __init__(self, M: int, B: float):
        """
        Initialize NodesStructure with parameters ``M`` and ``B``.

        :param M: maximum block size. Blocks exceeding ``M`` will be split.
        :param B: upper bound on all values of D1.
        """
        self.M = M
        self.B = B
        self.D0 = _BlockSequence()
        block = _Block(B)
        self.D1 = _BlockSequence(block)
        self.bst = SortedKeyList([(B, block)], key = lambda x : x[0])
        self.map: Dict[int, _Item] = {}

    def insert(self, key: int, value: float):
        """
        Insert or update key-value pair in D1.

        If key already exists with higher value, replaces it. Otherwise, finds appropriate D1
        block via BST and appends item. Triggers split if block exceeds M items.

        Complexity:
            O(max(1, M/B) * log(B/M)) amortized time.

        :param key: the node to insert.
        :param value: the distance of the node
        """
        if key in self.map:
            if self.map[key].value <= value:
                return
            block = self.map[key].block
            block.delete(self.map[key])
            if block.size == 0:
                self._delete_block(block)
        item = _Item(key, value)
        if not self.bst:
            block = _Block(self.B)
            self.bst.add((self.B, block))

        index = self.bst.bisect_left((value, None))
        assert index < len(self.bst), "Inserting value bigger than upper bound"
        block: _Block = self.bst[index][1]
        block.append(item)
        if block.size > self.M:
            left, right = block.split()
            del self.bst[index]
            self.bst.add((right.ub, right))
            self.bst.add((left.ub, left))
            self.D1.substitute_block_with_split(block, (left, right))
        self.map[key] = item

    def batch_prepend(self, block: Set[Tuple[int, float]]):
        """
        Batch prepend set of key-value pairs to front of D0.

        Creates new block from items, handles duplicates with existing keys, then prepends to D0.

        Complexity:
            O(|items| * max(1, M/B) * log(B/M)) amortized time.

        :param block: set of (``key``, ``value``) items to prepend.
        """
        if not block:
            return
        _block = _Block()
        for elem in block:
            item = _Item(elem[0], elem[1])
            _block.append(item)
        self._batch_prepend(_block)

    def _batch_prepend(self, block: _Block):
        """
        Internal batch prepend handling duplicate resolution.

        :param block: the block containing the items to prepend.
        """
        for item in block:
            if item.key in self.map:
                if self.map[item.key].value <= item.value:
                    block.delete(item)
                    continue
                self.map[item.key].block.delete(self.map[item.key])
            self.map[item.key] = item
        if block.size == 0:
            return
        self.D0.batch_prepend(block, self.M)

    def pull(self):
        """
        Extract M vertices with smallest values and return boundary.

        Collects items from D0 and D1 until M items found or structure emptied. Uses
        median-finding if more than M candidates to identify exact cutoff. Removes extracted
        items from structure and map.

         Complexity:
            O(M) amortized time

        :return: a tuple (``upper_bound``, ``item_set``), where ``upper_bound`` is the value that
        divides the extracted block from the remaining (if any, otherwise is the upper bound of
        the structure), and ``item_set`` is the list of extracted items.
        """
        assert not self.is_empty(), "Can't pull from empty structure"
        candidates: List[_Item] = []
        count_0 = count_1 = 0
        for block in self.D0:
            count_0 += block.size
            candidates.extend(block)
            if count_0 >= self.M:
                break
        for block in self.D1:
            count_1 += block.size
            candidates.extend(block)
            if count_1 >= self.M:
                break
        if len(candidates) <= self.M:
            return self._create_set(candidates, self.B)
        items: List[_Item] = []
        extra_items: List[_Item] = []
        remaining_items: List[_Item] = []
        index = select_fast(candidates, self.M)
        for item in candidates:
            if item.value < candidates[index].value:
                items.append(item)
            elif item.value == candidates[index].value:
                extra_items.append(item)
            else:
                remaining_items.append(item)
        extra_items_num = self.M - len(items)
        items.extend(extra_items[:extra_items_num])
        remaining_items.extend(extra_items[extra_items_num:])
        ub_index = select_fast(remaining_items, 0)
        return self._create_set(items, remaining_items[ub_index].value)

    def _create_set(self, items: List[_Item], upper_bound: float):
        """
        Helper to create return value for pull operation.

        :param items: the items to extract from the structure.
        :param upper_bound: the upper bound to reproduce in output.
        :return: a tuple (``upper_bound``, ``item_set``), output of ``pull()``.
        """
        item_set: Set[int] = set()
        for item in items:
            self.map.pop(item.key)
            orig_block = item.block
            orig_block.remove(item)
            if orig_block.size == 0:
                self._delete_block(orig_block)
            item_set.add(item.key)
        return upper_bound, item_set

    def _delete_block(self, block: _Block):
        """
        Delete block from its BlockSequence.

        :param block: the block to delete.
        """
        assert block.size == 0, "Can't delete a non-empty block"
        if block.ub is not None:
            if block is self.bst[-1][1]:
                return
            self.bst.remove((block.ub, block))
            self.D1.delete(block)
        else:
            self.D0.delete(block)

    def is_empty(self):
        """
        Check if the structure is empty (D0 is empty and D1 has a single empty block).

        :return: ``True`` if the structure is empty, ``False`` otherwise.
        """
        return self.D0.is_empty() and self.D1.size == 1 and self.D1.head.is_empty()
