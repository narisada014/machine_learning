# A. 1
n = input()

a = list(map(int, input().split()))

arr_length = len(a)
alice_point = 0
bob_point = 0

for _ in range(arr_length):
    if len(a) is 0:
        break
    alice_max = max(a)
    alice_point += int(max(a))
    a.remove(alice_max)

    if len(a) is 0:
        break
    bob_max = max(a)
    bob_point += int(max(a))
    a.remove(bob_max)


print(alice_point - bob_point)
# メモリ使いすぎ

# A. 2
n = int(input())
# 標準入力される数値をスプリットと同時にintegerにしてlistにする
# そのlistをsortしてreverseしてやる。sortだと小さい順番になっているのでreverse
a = sorted(list(map(int, input().split())), reverse=True)
# 与えられた配列の偶数番目の合計と奇数番目の合計の差を計算している
print(sum(a[::2]) - sum(a[1::2]))
