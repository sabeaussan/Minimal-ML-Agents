from collections import defaultdict
from collections.abc import MutableMapping
import enum
import itertools
from typing import BinaryIO, DefaultDict, List, Tuple, Union, Optional

import numpy as np
import h5py

from mlagents_envs.exception import UnityException

# Elements in the buffer can be np.ndarray, or in the case of teammate obs, actions, rewards,
# a List of np.ndarray. This is done so that we don't have duplicated np.ndarrays, only references.
BufferEntry = Union[np.ndarray, List[np.ndarray]]


class BufferException(UnityException):
    """
    Related to errors with the Buffer.
    """

    pass


class BufferKey(enum.Enum):
    CONTINUOUS_ACTION = "continuous_action"
    NEXT_CONT_ACTION = "next_continuous_action"
    CONTINUOUS_LOG_PROBS = "continuous_log_probs"
    DONE = "done"
    ENVIRONMENT_REWARDS = "environment_rewards"
    PREV_ACTION = "prev_action"

    ADVANTAGES = "advantages"
    DISCOUNTED_RETURNS = "discounted_returns"



class ObservationKeyPrefix(enum.Enum):
    OBSERVATION = "obs"
    NEXT_OBSERVATION = "next_obs"


class RewardSignalKeyPrefix(enum.Enum):
    # Reward signals
    REWARDS = "rewards"
    VALUE_ESTIMATES = "value_estimates"
    RETURNS = "returns"
    ADVANTAGE = "advantage"
    BASELINES = "baselines"


AgentBufferKey = Union[
    BufferKey, Tuple[ObservationKeyPrefix, int], Tuple[RewardSignalKeyPrefix, str]
]


class RewardSignalUtil:
    @staticmethod
    def rewards_key(name: str) -> AgentBufferKey:
        return RewardSignalKeyPrefix.REWARDS, name

    @staticmethod
    def value_estimates_key(name: str) -> AgentBufferKey:
        return RewardSignalKeyPrefix.RETURNS, name

    @staticmethod
    def returns_key(name: str) -> AgentBufferKey:
        return RewardSignalKeyPrefix.RETURNS, name

    @staticmethod
    def advantage_key(name: str) -> AgentBufferKey:
        return RewardSignalKeyPrefix.ADVANTAGE, name

    @staticmethod
    def baseline_estimates_key(name: str) -> AgentBufferKey:
        return RewardSignalKeyPrefix.BASELINES, name


class AgentBufferField(list):
    """
    AgentBufferField is a list of numpy arrays, or List[np.ndarray] for group entries.
    When an agent collects a field, you can add it to its AgentBufferField with the append method.
    """

    def __init__(self, *args, **kwargs):
        self.padding_value = 0
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return f"AgentBufferField: {super().__str__()}"

    def __getitem__(self, index):
        return_data = super().__getitem__(index)
        if isinstance(return_data, list):
            return AgentBufferField(return_data)
        else:
            return return_data

    @property
    def contains_lists(self) -> bool:
        """
        Checks whether this AgentBufferField contains List[np.ndarray].
        """
        return len(self) > 0 and isinstance(self[0], list)

    def append(self, element: BufferEntry, padding_value: float = 0.0) -> None:
        """
        Adds an element to this list. Also lets you change the padding
        type, so that it can be set on append (e.g. action_masks should
        be padded with 1.)
        :param element: The element to append to the list.
        :param padding_value: The value used to pad when get_batch is called.
        """
        super().append(element)
        self.padding_value = padding_value

    def set(self, data: List[BufferEntry]) -> None:
        """
        Sets the list of BufferEntry to the input data
        :param data: The BufferEntry list to be set.
        """
        self[:] = []
        self[:] = data

    def get_batch(
        self,
        batch_size: int = None,
        training_length: Optional[int] = 1,
        sequential: bool = True,
    ) -> List[BufferEntry]:
        """
        Retrieve the last batch_size elements of length training_length
        from the list of np.array
        :param batch_size: The number of elements to retrieve. If None:
        All elements will be retrieved.
        :param training_length: The length of the sequence to be retrieved. If
        None: only takes one element.
        :param sequential: If true and training_length is not None: the elements
        will not repeat in the sequence. [a,b,c,d,e] with training_length = 2 and
        sequential=True gives [[0,a],[b,c],[d,e]]. If sequential=False gives
        [[a,b],[b,c],[c,d],[d,e]]
        """
        return self

    def reset_field(self) -> None:
        """
        Resets the AgentBufferField
        """
        self[:] = []


class AgentBuffer(MutableMapping):
    """
    AgentBuffer contains a dictionary of AgentBufferFields. Each agent has his own AgentBuffer.
    The keys correspond to the name of the field. Example: state, action
    """

    # Whether or not to validate the types of keys at runtime
    # This should be off for training, but enabled for testing
    CHECK_KEY_TYPES_AT_RUNTIME = False

    def __init__(self):
        self.last_brain_info = None
        self.last_take_action_outputs = None
        self._fields: DefaultDict[AgentBufferKey, AgentBufferField] = defaultdict(
            AgentBufferField
        )

    def __str__(self):
        return ", ".join(
            ["'{}' : {}".format(k, str(self[k])) for k in self._fields.keys()]
        )

    def reset_agent(self) -> None:
        """
        Resets the AgentBuffer
        """
        for f in self._fields.values():
            f.reset_field()
        self.last_brain_info = None
        self.last_take_action_outputs = None

    @staticmethod
    def _check_key(key):
        if isinstance(key, BufferKey):
            return
        if isinstance(key, tuple):
            key0, key1 = key
            if isinstance(key0, ObservationKeyPrefix):
                if isinstance(key1, int):
                    return
                raise KeyError(f"{key} has type ({type(key0)}, {type(key1)})")
            if isinstance(key0, RewardSignalKeyPrefix):
                if isinstance(key1, str):
                    return
                raise KeyError(f"{key} has type ({type(key0)}, {type(key1)})")
        raise KeyError(f"{key} is a {type(key)}")

    @staticmethod
    def _encode_key(key: AgentBufferKey) -> str:
        """
        Convert the key to a string representation so that it can be used for serialization.
        """
        if isinstance(key, BufferKey):
            return key.value
        prefix, suffix = key
        return f"{prefix.value}:{suffix}"

    @staticmethod
    def _decode_key(encoded_key: str) -> AgentBufferKey:
        """
        Convert the string representation back to a key after serialization.
        """
        # Simple case: convert the string directly to a BufferKey
        try:
            return BufferKey(encoded_key)
        except ValueError:
            pass

        # Not a simple key, so split into two parts
        prefix_str, _, suffix_str = encoded_key.partition(":")

        # See if it's an ObservationKeyPrefix first
        try:
            return ObservationKeyPrefix(prefix_str), int(suffix_str)
        except ValueError:
            pass

        # If not, it had better be a RewardSignalKeyPrefix
        try:
            return RewardSignalKeyPrefix(prefix_str), suffix_str
        except ValueError:
            raise ValueError(f"Unable to convert {encoded_key} to an AgentBufferKey")

    def __getitem__(self, key: AgentBufferKey) -> AgentBufferField:
        if self.CHECK_KEY_TYPES_AT_RUNTIME:
            self._check_key(key)
        return self._fields[key]

    def __setitem__(self, key: AgentBufferKey, value: AgentBufferField) -> None:
        if self.CHECK_KEY_TYPES_AT_RUNTIME:
            self._check_key(key)
        self._fields[key] = value

    def __delitem__(self, key: AgentBufferKey) -> None:
        if self.CHECK_KEY_TYPES_AT_RUNTIME:
            self._check_key(key)
        self._fields.__delitem__(key)

    def __iter__(self):
        return self._fields.__iter__()

    def __len__(self) -> int:
        return self._fields.__len__()

    def __contains__(self, key):
        if self.CHECK_KEY_TYPES_AT_RUNTIME:
            self._check_key(key)
        return self._fields.__contains__(key)

    def check_length(self, key_list: List[AgentBufferKey]) -> bool:
        """
        Some methods will require that some fields have the same length.
        check_length will return true if the fields in key_list
        have the same length.
        :param key_list: The fields which length will be compared
        """
        if self.CHECK_KEY_TYPES_AT_RUNTIME:
            for k in key_list:
                self._check_key(k)

        if len(key_list) < 2:
            return True
        length = None
        for key in key_list:
            if key not in self._fields:
                return False
            if (length is not None) and (length != len(self[key])):
                return False
            length = len(self[key])
        return True

    def shuffle(
        self, key_list: List[AgentBufferKey] = None
    ) -> None:
        """
        Shuffles the fields in key_list in a consistent way: The reordering will
        be the same across fields.
        :param key_list: The fields that must be shuffled.
        """
        if key_list is None:
            key_list = list(self._fields.keys())
        if not self.check_length(key_list):
            raise BufferException(
                "Unable to shuffle if the fields are not of same length"
            )
        s = np.arange(len(self[key_list[0]]))
        np.random.shuffle(s)
        for key in key_list:
            buffer_field = self[key]
            tmp: List[np.ndarray] = []
            for i in s:
                tmp += buffer_field[i : (i + 1)]
            buffer_field.set(tmp)

    def make_mini_batch(self, start: int, end: int) -> "AgentBuffer":
        """
        Creates a mini-batch from buffer.
        :param start: Starting index of buffer.
        :param end: Ending index of buffer.
        :return: Dict of mini batch.
        """
        mini_batch = AgentBuffer()
        for key, field in self._fields.items():
            # slicing AgentBufferField returns a List[Any}
            mini_batch[key] = field[start:end]  # type: ignore
        return mini_batch

    def sample_mini_batch(self, batch_size: int) -> "AgentBuffer":
        """
        Creates a mini-batch from a random start and end.
        :param batch_size: number of elements to withdraw.
        :param sequence_length: Length of sequences to sample.
            Number of sequences to sample will be batch_size/sequence_length.
        """
        mini_batch = AgentBuffer()
        buff_len = self.num_experiences
        start_idxes = (
            np.random.randint(buff_len, size=batch_size)
        )  # Sample random sequence starts
        for key in self:
            buffer_field = self[key]
            mb_list = (buffer_field[i : i + 1] for i in start_idxes)
            # See comparison of ways to make a list from a list of lists here:
            # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
            mini_batch[key].set(list(itertools.chain.from_iterable(mb_list)))
        return mini_batch

    def save_to_file(self, file_object: BinaryIO) -> None:
        """
        Saves the AgentBuffer to a file-like object.
        """
        with h5py.File(file_object, "w") as write_file:
            for key, data in self.items():
                write_file.create_dataset(
                    self._encode_key(key), data=data, dtype="f", compression="gzip"
                )

    def load_from_file(self, file_object: BinaryIO) -> None:
        """
        Loads the AgentBuffer from a file-like object.
        """
        with h5py.File(file_object, "r") as read_file:
            for key in list(read_file.keys()):
                decoded_key = self._decode_key(key)
                self[decoded_key] = AgentBufferField()
                # extend() will convert the numpy array's first dimension into list
                self[decoded_key].extend(read_file[key][()])

    def truncate(self, max_length: int, sequence_length: int = 1) -> None:
        """
        Truncates the buffer to a certain length.

        This can be slow for large buffers. We compensate by cutting further than we need to, so that
        we're not truncating at each update. Note that we must truncate an integer number of sequence_lengths
        param: max_length: The length at which to truncate the buffer.
        """
        current_length = self.num_experiences
        # make max_length an integer number of sequence_lengths
        max_length -= max_length % sequence_length
        if current_length > max_length:
            for _key in self.keys():
                self[_key][:] = self[_key][current_length - max_length :]

    def resequence_and_append(
        self,
        target_buffer: "AgentBuffer",
        key_list: List[AgentBufferKey] = None,
        batch_size: int = None,
        training_length: int = None,
    ) -> None:
        """
        Takes in a batch size and training length (sequence length), and appends this AgentBuffer to target_buffer
        properly padded for LSTM use. Optionally, use key_list to restrict which fields are inserted into the new
        buffer.
        :param target_buffer: The buffer which to append the samples to.
        :param key_list: The fields that must be added. If None: all fields will be appended.
        :param batch_size: The number of elements that must be appended. If None: All of them will be.
        :param training_length: The length of the samples that must be appended. If None: only takes one element.
        """
        if key_list is None:
            key_list = list(self.keys())
        if not self.check_length(key_list):
            raise BufferException(
                f"The length of the fields {key_list} were not of same length"
            )
        for field_key in key_list:
            target_buffer[field_key].extend(self[field_key])

    @property
    def num_experiences(self) -> int:
        """
        The number of agent experiences in the AgentBuffer, i.e. the length of the buffer.

        An experience consists of one element across all of the fields of this AgentBuffer.
        Note that these all have to be the same length, otherwise shuffle and append_to_update_buffer
        will fail.
        """
        if self.values():
            return len(next(iter(self.values())))
        else:
            return 0
